from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER, HANDSHAKE_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_3
from os_ken.lib.packet import packet
from os_ken.lib.packet import ethernet, arp, ipv4, ether_types
from os_ken.controller import ofp_event
from os_ken.topology import event
import sys
from network_awareness import NetworkAwareness
import networkx as nx
ETHERNET = ethernet.ethernet.__name__
ETHERNET_MULTICAST = "ff:ff:ff:ff:ff:ff"
ARP = arp.arp.__name__
class ShortestDelay(app_manager.OSKenApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {
        'network_awareness': NetworkAwareness
    }

    def __init__(self, *args, **kwargs):
        super(ShortestDelay, self).__init__(*args, **kwargs)
        self.network_awareness = kwargs['network_awareness']
        self.weight = 'delay' # do not forget to change to 'delay' if you want to use delay
        self.mac_to_port = {}
        self.sw = {}
        self.path=None

    def add_flow(self, datapath, priority, match, actions, idle_timeout=0, hard_timeout=0):
        dp = datapath
        ofp = dp.ofproto
        parser = dp.ofproto_parser

        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(
            datapath=dp, priority=priority,
            idle_timeout=idle_timeout,
            hard_timeout=hard_timeout,
            match=match, instructions=inst)
        dp.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        dp = msg.datapath
        ofp = dp.ofproto
        parser = dp.ofproto_parser

        dpid = dp.id
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth_pkt = pkt.get_protocol(ethernet.ethernet)
        arp_pkt = pkt.get_protocol(arp.arp)
        ipv4_pkt = pkt.get_protocol(ipv4.ipv4)

        pkt_type = eth_pkt.ethertype

        # layer 2 self-learning
        dst_mac = eth_pkt.dst
        src_mac = eth_pkt.src


        if isinstance(arp_pkt, arp.arp):
            self.handle_arp(msg, in_port, dst_mac,src_mac, pkt,pkt_type)

        if isinstance(ipv4_pkt, ipv4.ipv4):
            self.handle_ipv4(msg, ipv4_pkt.src, ipv4_pkt.dst, pkt_type)

    def handle_arp(self, msg, in_port, dst,src, pkt,pkt_type):
        """
        使用 (dpid, src_mac, dst_mac) -> in_port 的方法处理 ARP 环路
        """
        datapath = msg.datapath
        dpid = datapath.id
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto
        
        # 构造唯一键
        key = (dpid, src, dst)
        
        # 检测环路
        if key in self.sw:
            if self.sw[key] != in_port:
                # 环路！丢弃包
                self.logger.info(
                    "[ARP loop detected]\ndpid=%s, src=%s, dst=%s, "
                    "in_port=%s (previous=%s)",
                    dpid, src, dst, in_port, self.sw[key]
                )
                return  # 丢弃包
        else:
            # 首次记录
            self.sw[key] = in_port
        
        # 洪泛 ARP
        actions = [parser.OFPActionOutput(ofproto.OFPP_FLOOD)]
        out = parser.OFPPacketOut(
            datapath=datapath, 
            buffer_id=msg.buffer_id, 
            in_port=in_port, 
            actions=actions, 
            data=msg.data
        )
        datapath.send_msg(out)

    def handle_ipv4(self, msg, src_ip, dst_ip, pkt_type):
        parser = msg.datapath.ofproto_parser

        dpid_path = self.network_awareness.shortest_path(src_ip, dst_ip,weight=self.weight)
        if not dpid_path:
            return
        
        self.path=dpid_path
        # get port path:  h1 -> in_port, s1, out_port -> h2
        port_path = []
        for i in range(1, len(dpid_path) - 1):
            in_port = self.network_awareness.link_info[(dpid_path[i], dpid_path[i - 1])]
            out_port = self.network_awareness.link_info[(dpid_path[i], dpid_path[i + 1])]
            port_path.append((in_port, dpid_path[i], out_port))
        self.show_path(src_ip, dst_ip, port_path)

        # calc path delay and print
        # style:
        #   "delay = %.5fms"
        #   "time = %.5fms"
        '''
            利用dpid_path(最短路)和link_delay_table计算path delay, path RTT
            输出link delay dict, path delay， path RTT
            输出语句示例:
            self.logger.info('link delay dict: %s', )
            self.logger.info('path delay = %.5fms', )
            self.logger.info('path RTT = %.5fms', )
        '''

        # ========== 新增：计算路径时延 ==========
        # 构建链路时延字典
        link_delay_dict = {}
        path_delay = 0.0
        
        for i in range(1, len(dpid_path) - 1):
            src = dpid_path[i]
            dst = dpid_path[i + 1]
            
            # 从拓扑图中获取时延
            if self.network_awareness.topo_map.has_edge(src, dst):
                delay = self.network_awareness.topo_map[src][dst].get('delay', 0)
                link_delay_dict[f"s{src}->s{dst}"] = delay * 1000  # 转换为 ms
                path_delay += delay
        
        # 计算 Path RTT（往返时间 = 2 * 单向时延）
        path_RTT = path_delay * 2
        
        # 输出结果
        self.logger.info('link delay dict: %s', link_delay_dict)
        self.logger.info("path delay = %.5fms", path_delay * 1000)
        self.logger.info("path RTT = %.5fms", path_RTT * 1000)

        # send flow mod
        for node in port_path:
            in_port, dpid, out_port = node
            self.send_flow_mod(parser, dpid, pkt_type, src_ip, dst_ip, in_port, out_port)
            self.send_flow_mod(parser, dpid, pkt_type, dst_ip, src_ip, out_port, in_port)

        # send packet_out
        _, dpid, out_port = port_path[-1]
        dp = self.network_awareness.switch_info[dpid]
        ofp = dp.ofproto
        actions = [parser.OFPActionOutput(out_port)]
        out = parser.OFPPacketOut(
            datapath=dp, buffer_id=msg.buffer_id, in_port=ofp.OFPP_CONTROLLER, actions=actions, data=msg.data)
        dp.send_msg(out)

    def send_flow_mod(self, parser, dpid, pkt_type, src_ip, dst_ip, in_port, out_port):
        dp = self.network_awareness.switch_info[dpid]
        match = parser.OFPMatch(
            in_port=in_port, eth_type=pkt_type, ipv4_src=src_ip, ipv4_dst=dst_ip)
        actions = [parser.OFPActionOutput(out_port)]
        self.add_flow(dp, 1, match, actions, 10, 30)

    def show_path(self, src, dst, port_path):
        self.logger.info('path: {} -> {}'.format(src, dst))
        path = src + ' -> '
        for node in port_path:
            path += '{}:s{}:{}'.format(*node) + ' -> '
        path += dst
        self.logger.info(path)

    
