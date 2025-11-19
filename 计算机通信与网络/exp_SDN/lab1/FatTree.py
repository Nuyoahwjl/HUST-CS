#!/usr/bin/env python3

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import OVSSwitch, Controller
from mininet.cli import CLI
from mininet.log import setLogLevel, info

class FatTreeTopo(Topo):
    def __init__(self, k=4, **opts):
        # 先设置k值
        self.k = k
        self.pod = k
        
        # 初始化列表
        self.core_switches = []
        self.aggregation_switches = []
        self.edge_switches = []
        self.host_list = [] 
        
        # 主机计数器
        self.host_counter = 1
        
        # 然后调用父类的初始化
        super(FatTreeTopo, self).__init__(**opts)
    
    def build(self):
        """构建FatTree拓扑 - 这个方法会被Mininet自动调用"""
        # 创建核心交换机 (k/2)^2 个 - 命名为 c1, c2, c3...
        core_count = (self.k // 2) ** 2
        for i in range(core_count):
            sw = self.addSwitch(f'c{i+1}')
            self.core_switches.append(sw)
        
        # 为每个pod创建聚合交换机和边缘交换机
        for pod in range(self.pod):
            # 每个pod有k/2个聚合交换机 - 命名为 a11, a12, a21, a22...
            for agg in range(self.k // 2):
                agg_sw = self.addSwitch(f'a{pod+1}{agg+1}')
                self.aggregation_switches.append(agg_sw)
            
            # 每个pod有k/2个边缘交换机 - 命名为 e11, e12, e21, e22...
            for edge in range(self.k // 2):
                edge_sw = self.addSwitch(f'e{pod+1}{edge+1}')
                self.edge_switches.append(edge_sw)
                
                # 每个边缘交换机连接k/2个主机 - 命名为 h1, h2, h3...
                for host_num in range(self.k // 2):
                    host = self.addHost(f'h{self.host_counter}')
                    self.host_list.append(host)
                    self.addLink(edge_sw, host)
                    self.host_counter += 1
        
        # 连接核心层和聚合层
        self.connect_core_to_aggregation()
        
        # 连接聚合层和边缘层（在同一个pod内）
        self.connect_aggregation_to_edge()
    
    def connect_core_to_aggregation(self):
        """连接核心交换机和聚合交换机"""
        core_per_group = self.k // 2
        
        for core_index, core_sw in enumerate(self.core_switches):
            core_group = core_index // core_per_group  # 核心交换机所属的组
            
            for pod in range(self.pod):
                # 每个核心交换机连接到每个pod中的特定聚合交换机
                agg_index_in_pod = pod * (self.k // 2) + core_group
                if agg_index_in_pod < len(self.aggregation_switches):
                    agg_sw = self.aggregation_switches[agg_index_in_pod]
                    self.addLink(core_sw, agg_sw)
    
    def connect_aggregation_to_edge(self):
        """连接聚合交换机和边缘交换机（在同一个pod内）"""
        switches_per_pod = self.k
        total_pods = self.pod
        
        for pod in range(total_pods):
            # 获取当前pod的所有聚合交换机
            agg_start = pod * (self.k // 2)
            agg_end = agg_start + (self.k // 2)
            pod_agg_switches = self.aggregation_switches[agg_start:agg_end]
            
            # 获取当前pod的所有边缘交换机
            edge_start = pod * (self.k // 2)
            edge_end = edge_start + (self.k // 2)
            pod_edge_switches = self.edge_switches[edge_start:edge_end]
            
            # 连接pod内的所有聚合交换机和所有边缘交换机
            for agg_sw in pod_agg_switches:
                for edge_sw in pod_edge_switches:
                    self.addLink(agg_sw, edge_sw)


def show_MAC_tables(net:Mininet):
    """显示所有交换机的MAC表"""
    info("*** Analyzing data paths using ovs-appctl fdb/show\n")
    for switch in net.switches:
        info(f"--- {switch.name:3} MAC table ---\n")
        try:
            output = switch.cmd(f"ovs-appctl fdb/show {switch.name}")
            info(output)
        except:
            info(f"Could not get MAC table for {switch.name:3}\n")

def run():
    """启动FatTree拓扑"""
    # 创建拓扑
    topo = FatTreeTopo(k=4)
    
    # 创建网络，不使用控制器
    net = Mininet(topo=topo, switch=OVSSwitch, controller=None, autoSetMacs=True)
    
    info('*** 启动网络\n')
    net.start()
    
    info('*** 配置交换机standalone模式和STP\n')
    for switch in net.switches:
        # Set fail-mode to standalone
        switch.cmd('ovs-vsctl set-fail-mode', switch.name, 'standalone')
        # Enable Spanning Tree Protocol to prevent broadcast storms
        switch.cmd('ovs-vsctl set Bridge', switch.name, 'stp_enable=true')
        info(f"--- {switch.name:3}: standalone mode + STP enabled\n")
    info("*** Waiting for STP to converge (30 seconds)...\n")
    from time import sleep
    sleep(30)
    
    info('*** 测试网络连通性\n')
    res = net.pingAll()
    if res == 0:
       info("*** SUCCESS: All hosts are connected!\n")
       show_MAC_tables(net)
    else:
       info("*** FAILURE: Some hosts are not connected\n")
       info("*** Use Wireshark to analyze packet flow and debug connectivity\n")
        
    # info('*** Ping h1 和 h16\n')
    # h1 = net.get('h1')
    # h16 = net.get('h16')
    # result = h1.cmd('ping -c 4 %s' % h16.IP())
    # info(result)
    # show_MAC_tables(net)
    
    info('*** 运行CLI\n')
    CLI(net)
    
    info('*** 停止网络\n')
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    run()