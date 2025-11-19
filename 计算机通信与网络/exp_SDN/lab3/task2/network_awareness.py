from os_ken.base import app_manager
from os_ken.base.app_manager import lookup_service_brick
from os_ken.ofproto import ofproto_v1_3
from os_ken.controller.handler import set_ev_cls
from os_ken.controller.handler import MAIN_DISPATCHER, CONFIG_DISPATCHER, DEAD_DISPATCHER
from os_ken.controller import ofp_event
from os_ken.lib.packet import packet
from os_ken.lib.packet import ethernet, arp
from os_ken.lib import hub
from os_ken.topology import event
from os_ken.topology.api import get_all_host, get_all_link, get_all_switch
from os_ken.topology.switches import LLDPPacket
from os_ken.lib import hub

import networkx as nx
import copy
import time


GET_TOPOLOGY_INTERVAL = 2
SEND_ECHO_REQUEST_INTERVAL = .05
GET_DELAY_INTERVAL = 2


class NetworkAwareness(app_manager.OSKenApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(NetworkAwareness, self).__init__(*args, **kwargs)
        self.switch_info = {}  # dpid: datapath
        self.link_info = {}  # (s1, s2): s1.port
        self.port_link={} # s1,port:s1,s2
        self.port_info = {}  # dpid: (ports linked hosts)
        self.topo_map = nx.Graph()
        self.topo_thread = hub.spawn(self._get_topology)

        self.weight = 'delay' # don't forget change it to 'delay'
        # add your variables here
        self.lldp_delay_table = {}     # (src_dpid, dst_dpid) -> T_lldp
        self.switches = {}             # switches 实例
        self.echo_RTT_table = {}       # dpid -> T_echo
        self.echo_send_timestamp = {}  # dpid -> send_time
        self.link_delay_table = {}     # (dpid1, dpid2) -> delay

        # 启动 Echo 测量线程
        self.echo_thread = hub.spawn(self.examine_echo_RTT)

    def add_flow(self, datapath, priority, match, actions):
        dp = datapath
        ofp = dp.ofproto
        parser = dp.ofproto_parser

        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=dp, priority=priority, match=match, instructions=inst)
        dp.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        msg = ev.msg
        dp = msg.datapath
        ofp = dp.ofproto
        parser = dp.ofproto_parser

        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)]
        self.add_flow(dp, 0, match, actions)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def state_change_handler(self, ev):
        dp = ev.datapath
        dpid = dp.id

        if ev.state == MAIN_DISPATCHER:
            self.switch_info[dpid] = dp

        if ev.state == DEAD_DISPATCHER:
            del self.switch_info[dpid]
    def _get_topology(self):
        _hosts, _switches, _links = None, None, None
        while True:
            hosts = get_all_host(self)
            switches = get_all_switch(self)
            links = get_all_link(self)

            # update topo_map when topology change
            if [str(x) for x in hosts] == _hosts and [str(x) for x in switches] == _switches and [str(x) for x in links] == _links:
                hub.sleep(GET_TOPOLOGY_INTERVAL)
                continue
            _hosts, _switches, _links = [str(x) for x in hosts], [str(x) for x in switches], [str(x) for x in links]

            for switch in switches:
                self.port_info.setdefault(switch.dp.id, set())
                # record all ports
                for port in switch.ports:
                    self.port_info[switch.dp.id].add(port.port_no)

            for host in hosts:
                # take one ipv4 address as host id
                if host.ipv4:
                    self.link_info[(host.port.dpid, host.ipv4[0])] = host.port.port_no
                    self.topo_map.add_edge(host.ipv4[0], host.port.dpid, hop=1, delay=0, is_host=True)

            for link in links:
                # delete ports linked switches or hosts
                self.port_info[link.src.dpid].discard(link.src.port_no)
                self.port_info[link.dst.dpid].discard(link.dst.port_no)

                # s1 -> s2: s1.port, s2 -> s1: s2.port
                self.port_link[(link.src.dpid,link.src.port_no)]=(link.src.dpid, link.dst.dpid)
                self.port_link[(link.dst.dpid,link.dst.port_no)] = (link.dst.dpid, link.src.dpid)

                self.link_info[(link.src.dpid, link.dst.dpid)] = link.src.port_no
                self.link_info[(link.dst.dpid, link.src.dpid)] = link.dst.port_no

                # Calculate link delay
                '''
                TODO：
                	计算链路delay
                	将delay存入link_delay_table
                	使用self.logger.info打印delay消息
                '''
                delay = self.calculate_link_delay(link.src.dpid, link.dst.dpid)
                # 输出链路时延信息
                self.logger.info(
                    "Link: %s -> %s, delay: %.5fms",
                    link.src.dpid, link.dst.dpid, delay * 1000
                )
                # 添加边到拓扑图（包含 delay 属性）
                self.topo_map.add_edge(
                    link.src.dpid, link.dst.dpid, 
                    hop=1, 
                    delay=delay,  # 新增 delay 属性
                    is_host=False
                )

                # self.topo_map.add_edge(link.src.dpid, link.dst.dpid, hop=1, is_host=False)

            if self.weight == 'hop' or self.weight == 'delay':
                self.show_topo_map()
            hub.sleep(GET_TOPOLOGY_INTERVAL)

    def shortest_path(self, src, dst, weight='hop'):
        try:
            paths = list(nx.shortest_simple_paths(self.topo_map, src, dst, weight=weight))
            return paths[0]
        except:
            self.logger.info('host not find/no path')

    def show_topo_map(self):
        self.logger.info('topo map:')
        self.logger.info('{:^10s}  ->  {:^10s}'.format('node', 'node'))
        for src, dst in self.topo_map.edges:
            self.logger.info('{:^10s}      {:^10s}'.format(str(src), str(dst)))
        self.logger.info('\n')

    '''
        what you should do
        - add variables
        - lab3.1
            1. get lldp delay
            2. get echo delay
            3. calculate link delay
            4. get shortest path with networkx

        - lab3.2
            1. handle `EventOFPPortStatus`
            2. delete flow when port down
    '''


    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """处理 LLDP 消息，获取 LLDP 时延"""
        msg = ev.msg
        dpid = msg.datapath.id
        
        try:
            src_dpid, src_port_no = LLDPPacket.lldp_parse(msg.data)
            
            # 获取 switches 实例（只需获取一次）
            if not self.switches:
                self.switches = lookup_service_brick('switches')
            
            # 从 switches 中获取 LLDP 时延
            for port in self.switches.ports.keys():
                if src_dpid == port.dpid and src_port_no == port.port_no:
                    # 保存 T_lldp
                    self.lldp_delay_table[(src_dpid, dpid)] = \
                        self.switches.ports[port].delay
                    break
        except:
            return

    def send_echo_request(self, switch):
        """向交换机发送 Echo 请求"""
        datapath = switch.dp
        parser = datapath.ofproto_parser
        dpid = datapath.id
        
        # 记录发送时间
        send_time = time.time()
        self.echo_send_timestamp[dpid] = send_time
        
        # 构造 Echo 请求（data 必须是 bytes 类型）
        data = str(send_time).encode('utf-8')
        echo_req = parser.OFPEchoRequest(datapath, data=data)
        
        # 发送
        datapath.send_msg(echo_req)

    @set_ev_cls(ofp_event.EventOFPEchoReply, MAIN_DISPATCHER)
    def handle_echo_reply(self, ev):
        """处理 Echo 回复，计算 T_echo"""
        try:
            msg = ev.msg
            datapath = msg.datapath
            dpid = datapath.id
            
            # 记录接收时间
            recv_time = time.time()
            
            # 获取发送时间（可选：从 data 中解码验证）
            send_time = self.echo_send_timestamp.get(dpid)
            if send_time:
                # 计算 Echo RTT
                self.echo_RTT_table[dpid] = recv_time - send_time
        except Exception as e:
            self.logger.warning(f"Failed to handle echo reply: {e}")

    def examine_echo_RTT(self):
        """周期性测量 Echo RTT"""
        while True:
            # 获取所有交换机
            switches = get_all_switch(self)
            
            # 向每个交换机发送 Echo
            for switch in switches:
                self.send_echo_request(switch)
            
            # 睡眠（使用 hub.sleep 减少影响）
            hub.sleep(SEND_ECHO_REQUEST_INTERVAL)

    def calculate_link_delay(self, src_dpid, dst_dpid):
        """
        计算链路单向时延
        公式: delay = max((T_lldp12 + T_lldp21 - T_echo1 - T_echo2) / 2, 0)
        """
        try:
            # 获取 LLDP 往返时延
            lldp_12 = self.lldp_delay_table.get((src_dpid, dst_dpid))
            lldp_21 = self.lldp_delay_table.get((dst_dpid, src_dpid))
            
            # 获取 Echo RTT
            echo_1 = self.echo_RTT_table.get(src_dpid)
            echo_2 = self.echo_RTT_table.get(dst_dpid)

            # 计算链路时延
            delay = max((lldp_12 + lldp_21 - echo_1 - echo_2) / 2, 0.0)
            
            # 双向写缓存，后续可回退
            self.link_delay_table[(src_dpid, dst_dpid)] = delay
            self.link_delay_table[(dst_dpid, src_dpid)] = delay
            
            return delay
        
        except KeyError:
            # 链路发现和延迟计算是异步的，可能出现键不存在的情况
            return self.link_delay_table.get((src_dpid, dst_dpid), float('inf'))
        