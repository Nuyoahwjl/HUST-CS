from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER, CONFIG_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_3
from os_ken.lib.packet import packet
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import arp
from os_ken.lib.packet import ether_types

ETHERNET = ethernet.ethernet.__name__
ETHERNET_MULTICAST = "ff:ff:ff:ff:ff:ff"
ARP = arp.arp.__name__


class Switch_Dict(app_manager.OSKenApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(Switch_Dict, self).__init__(*args, **kwargs)
        self.sw = {} #(dpid, src_mac, dst_ip)=>in_port, you may use it in mission 2
        # maybe you need a global data structure to save the mapping
        # just data structure in mission 1
        # 维护 MAC 地址到端口的映射表: {dpid: {mac: port}}
        self.mac_to_port = {}
        
    def add_flow(self, datapath, priority, match, actions, idle_timeout=0, hard_timeout=0):
        dp = datapath
        ofp = dp.ofproto
        parser = dp.ofproto_parser
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=dp, priority=priority,
                                idle_timeout=idle_timeout,
                                hard_timeout=hard_timeout,
                                match=match, instructions=inst)
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

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        dp = msg.datapath
        ofp = dp.ofproto
        parser = dp.ofproto_parser

        # the identity of switch
        dpid = dp.id
        # the port that receive the packet
        in_port = msg.match['in_port']
        pkt = packet.Packet(msg.data)
        eth_pkt = pkt.get_protocol(ethernet.ethernet)
        if eth_pkt.ethertype == ether_types.ETH_TYPE_LLDP:
            return
        if eth_pkt.ethertype == ether_types.ETH_TYPE_IPV6:
            return
        # get the mac
        dst = eth_pkt.dst
        src = eth_pkt.src
        # get protocols
        header_list = dict((p.protocol_name, p) for p in pkt.protocols if type(p) != str)
        
        # 检测并防止 ARP 广播环路
        if dst == ETHERNET_MULTICAST and ARP in header_list:
        # you need to code here to avoid broadcast loop to finish mission 2
            arp_pkt = header_list[ARP]
            arp_dst_ip = arp_pkt.dst_ip
            
            # 构造映射键：(dpid, src_mac, dst_ip)
            key = (dpid, src, arp_dst_ip)
            
            # 检查是否已经记录过这个 ARP Request
            if key in self.sw:
                # 如果之前记录的端口与当前端口不同，说明产生了环路
                if self.sw[key] != in_port:
                    self.logger.info("[Dropping ARP request (Loop detected)]\ndpid=%s, src=%s, dst_ip=%s, in_port=%s (previous port=%s)", 
                                   dpid, src, arp_dst_ip, in_port, self.sw[key])
                    # 不执行任何转发操作，直接返回
                    return
            else:
                # 第一次收到这个 ARP Request，记录下来
                self.sw[key] = in_port
                self.logger.info("[Recording ARP request]\ndpid=%s, src=%s, dst_ip=%s, in_port=%s", 
                               dpid, src, arp_dst_ip, in_port)

        # self-learning
        # you need to code here to avoid the direct flooding
        # having fun
        # :)
        # just code in mission 1

        # 初始化该交换机的 MAC 表（如果还没有）
        self.mac_to_port.setdefault(dpid, {})
        
        # 学习源 MAC 地址和入端口的映射
        self.mac_to_port[dpid][src] = in_port
        
        # 查询目的 MAC 地址是否已学习
        if dst in self.mac_to_port[dpid]:
            # 映射表命中，获取输出端口
            out_port = self.mac_to_port[dpid][dst]
            
            # 打印五元组信息
            self.logger.info("[Packet matched]\ndpid=%s, src=%s, in_port=%s, dst=%s, out_port=%s", 
                           dpid, src, in_port, dst, out_port)
            
            # 构造匹配规则和动作
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst)
            actions = [parser.OFPActionOutput(out_port)]
            
            # 下发流表（可以修改 hard_timeout 参数来观察不同效果）
            # hard_timeout=0 表示永久有效
            # hard_timeout=5 表示 5 秒后失效
            self.add_flow(dp, 1, match, actions, hard_timeout=0)
            
            # 转发当前数据包
            out = parser.OFPPacketOut(
                datapath=dp,
                buffer_id=msg.buffer_id,
                in_port=in_port,
                actions=actions,
                data=msg.data
            )
            dp.send_msg(out)
        else:
            # 映射表未命中，洪泛
            actions = [parser.OFPActionOutput(ofp.OFPP_FLOOD)]
            out = parser.OFPPacketOut(
                datapath=dp,
                buffer_id=msg.buffer_id,
                in_port=in_port,
                actions=actions,
                data=msg.data
            )
            dp.send_msg(out)
