import warnings

warnings.filterwarnings("ignore")
from scapy.packet import Packet, bind_layers
from scapy.fields import ByteEnumField


class S7COMM(Packet):
    name = "S7COMM"
    fields_desc = [ByteEnumField("rosctr", 1, {1: "Job", 7: "UserData"})]


import csv
from scapy.all import *
from scapy.layers.l2 import ARP
from scapy.layers.inet import IP, TCP, UDP, Ether, ICMP
from scapy.layers.dns import DNS
from collections import defaultdict
import pandas as pd
import numpy as np
from scapy.contrib.modbus import ModbusADURequest, ModbusADUResponse
from datetime import datetime

bind_layers(TCP, S7COMM, dport=102)
bind_layers(TCP, S7COMM, sport=102)


def isok(_str):
    if (_str == '192.168.1.210'):
        return True
    elif (_str == '192.168.1.240'):
        return True
    return False


def str_ok(_str):
    if (_str.startswith('192.168.')):
        return True
    return False


def process_packet(pkt):
    try:
        # if pkt.haslayer(IP):
        #     if (str_ok(pkt[IP].src) == False):
        #         return
        #     if (str_ok(pkt[IP].dst) == False):
        #         return
        if pkt.haslayer(IP) and (pkt.haslayer(TCP) or pkt.haslayer(UDP)):
            modbus_func_code = 0
            modbus_trans_id = 0
            if ModbusADURequest in pkt:
                modbus_trans_id = pkt[ModbusADURequest].transId
                modbus_func_code = pkt[ModbusADURequest].funcCode
            elif ModbusADUResponse in pkt:
                modbus_trans_id = pkt[ModbusADUResponse].transId
                modbus_func_code = pkt[ModbusADUResponse].funcCode
            src_ip = pkt[IP].src
            dst_ip = pkt[IP].dst
            src_mac = pkt[IP].src
            dst_mac = pkt[IP].dst
            if pkt.haslayer(Ether):
                src_mac = pkt[Ether].src
                dst_mac = pkt[Ether].dst
            protocol = pkt[IP].proto
            src_port = pkt.sport
            dst_port = pkt.dport
            payload_len = len(pkt[IP].payload)

            flow_key = (
                src_ip, dst_ip, src_mac, dst_mac, protocol, src_port, dst_port, modbus_trans_id, modbus_func_code)
            flow_data = flow_dict[flow_key]

            # flow_data['protocol_name'] = 'Modbus/TCP' #if protocol == 6 else'UDP' if protocol == 17 else 'N/A'
            if protocol == 6:
                flow_data['protocol_name'] = 'TCP'
            elif protocol == 17:
                flow_data['protocol_name'] = 'UDP'
            if protocol == 1:
                flow_data['protocol_name'] = 'ICMP'
            if pkt.haslayer(DNS):
                flow_data['protocol_name'] = 'DNS'
            if protocol == 17 and pkt[UDP].dport == 137:
                flow_data['protocol_name'] = 'NBNS'
            if ModbusADURequest in pkt or ModbusADUResponse in pkt:
                flow_data['protocol_name'] = 'Modbus/TCP'
            if S7COMM in pkt:
                flow_data['protocol_name'] = 'S7COMM'

            flow_data['start_time'] = min(flow_data['start_time'], pkt.time)
            flow_data['end_time'] = max(flow_data['end_time'], pkt.time)

            if src_ip < dst_ip:
                flow_data['up_pkt_count'] += 1
                flow_data['up_payload_count'] += payload_len
                flow_data['up_payload_sizes'].append(payload_len)
                flow_data['up_pkt_times'].append(pkt.time)
            else:
                flow_data['down_pkt_count'] += 1
                flow_data['down_payload_count'] += payload_len
                flow_data['down_payload_sizes'].append(payload_len)
                flow_data['down_pkt_times'].append(pkt.time)

            print(pkt.time)


        # if pkt.haslayer(IP):
        #     if (str_ok(pkt[IP].src) == False):
        #         return
        #     if (str_ok(pkt[IP].dst) == False):
        #         return

        elif pkt.haslayer(ARP):
            modbus_func_code = 0
            modbus_trans_id = 0
            src_ip = pkt[ARP].psrc
            dst_ip = pkt[ARP].pdst
            src_mac = pkt[ARP].hwsrc
            dst_mac = pkt[ARP].hwdst
            if pkt.haslayer(Ether):
                src_mac = pkt[Ether].src
                dst_mac = pkt[Ether].dst
            # print(src_mac, "  ", dst_mac)
            src_port = 0
            dst_port = 0
            payload_len = 0
            flow_key = (src_ip, dst_ip, src_mac, dst_mac, 0, src_port, dst_port, modbus_trans_id, modbus_func_code)
            flow_data = flow_dict[flow_key]
            flow_data['protocol_name'] = 'ARP'
            # if pkt.haslayer(ARP):
            #     flow_data['protocol_name'] = 'ARP'
            #     src_ip = pkt[ARP].psrc
            #     dst_ip = pkt[ARP].pdst
            #     src_port = 0
            #     dst_port = 0
            #     payload_len = 0
            #     flow_key = (src_ip, dst_ip, 0, src_port, dst_port)
            #     flow_data = flow_dict[flow_key]
            flow_data['start_time'] = min(flow_data['start_time'], pkt.time)
            flow_data['end_time'] = max(flow_data['end_time'], pkt.time)

            if src_ip < dst_ip:
                flow_data['up_pkt_count'] += 1
                flow_data['up_payload_count'] += payload_len
                flow_data['up_payload_sizes'].append(payload_len)
                flow_data['up_pkt_times'].append(pkt.time)
            else:
                flow_data['down_pkt_count'] += 1
                flow_data['down_payload_count'] += payload_len
                flow_data['down_payload_sizes'].append(payload_len)
                flow_data['down_pkt_times'].append(pkt.time)

    except Exception as e:
        print("error")
        pass


def flow_statistics(flow_data):
    for key, value in flow_data.items():
        up_payload_sizes = np.array(value['up_payload_sizes'])
        down_payload_sizes = np.array(value['down_payload_sizes'])
        up_pkt_times = np.array(value['up_pkt_times'])
        down_pkt_times = np.array(value['down_pkt_times'])

        if len(up_payload_sizes) > 0:
            value['up_payload_mean'] = np.mean(up_payload_sizes)
            value['up_payload_min'] = np.min(up_payload_sizes)
            value['up_payload_max'] = np.max(up_payload_sizes)
            value['up_payload_std'] = np.std(up_payload_sizes)
        else:
            value['up_payload_mean'] = value['up_payload_min'] = value['up_payload_max'] = value['up_payload_std'] = 0

        if len(down_payload_sizes) > 0:
            value['down_payload_mean'] = np.mean(down_payload_sizes).astype(Decimal)
            value['down_payload_min'] = np.min(down_payload_sizes)
            value['down_payload_max'] = np.max(down_payload_sizes)
            value['down_payload_std'] = np.std(down_payload_sizes)
        else:
            value['down_payload_mean'] = value['down_payload_min'] = value['down_payload_max'] = value[
                'down_payload_std'] = 0

        if len(up_pkt_times) > 1:
            up_time_intervals = np.diff(up_pkt_times)
            value['up_time_interval_mean'] = np.mean(up_time_intervals)
            value['up_time_interval_min'] = np.min(up_time_intervals)
            value['up_time_interval_max'] = np.max(up_time_intervals)
            value['up_time_interval_std'] = np.std(up_time_intervals)
        else:
            value['up_time_interval_mean'] = 0
            value['up_time_interval_min'] = 0
            value['up_time_interval_max'] = 0
            value['up_time_interval_std'] = 0

        if len(down_pkt_times) > 1:
            down_time_intervals = np.diff(down_pkt_times)
            value['down_time_interval_mean'] = np.mean(down_time_intervals)
            value['down_time_interval_min'] = np.min(down_time_intervals)
            value['down_time_interval_max'] = np.max(down_time_intervals)
            value['down_time_interval_std'] = np.std(down_time_intervals)
            '''
            print(np.max(down_time_intervals))
            print(np.min(down_time_intervals))
            for i in range(0, len(down_time_intervals)):
                if down_time_intervals[i] == 0:
                    print(i)
            print('--------------------------------------')
            '''
        else:
            value['down_time_interval_mean'] = 0
            value['down_time_interval_min'] = 0
            value['down_time_interval_max'] = 0
            value['down_time_interval_std'] = 0


def write_to_csv():
    with open('111.csv', 'w', newline='') as csvfile:
        fieldnames = [
            'src_ip', 'dst_ip', 'src_mac', 'dst_mac', 'protocol_name', 'src_port', 'dst_port',
            'modbus_trans_id', 'modbus_func_code', 'up_pkt_count', 'down_pkt_count', 'up_payload_count',
            'down_payload_count', 'duration', 'up_payload_mean', 'down_payload_mean', 'up_payload_min',
            'down_payload_min', 'up_payload_max', 'down_payload_max', 'up_payload_std', 'down_payload_std',
            'up_time_interval_mean', 'down_time_interval_mean', 'up_time_interval_min', 'down_time_interval_min',
            'up_time_interval_max', 'down_time_interval_max', 'up_time_interval_std', 'down_time_interval_std'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for key, value in flow_dict.items():
            writer.writerow({
                'src_ip': key[0],
                'dst_ip': key[1],
                'src_mac': key[2],
                'dst_mac': key[3],
                'protocol_name': value['protocol_name'],
                'src_port': key[5],
                'dst_port': key[6],
                'modbus_trans_id': key[7],
                'modbus_func_code': key[8],
                'up_pkt_count': value['up_pkt_count'],
                'down_pkt_count': value['down_pkt_count'],
                'up_payload_count': value['up_payload_count'],
                'down_payload_count': value['down_payload_count'],
                'duration': value['end_time'] - value['start_time'],
                'up_payload_mean': value['up_payload_mean'],
                'down_payload_mean': value['down_payload_mean'],
                'up_payload_min': value['up_payload_min'],
                'down_payload_min': value['down_payload_min'],
                'up_payload_max': value['up_payload_max'],
                'down_payload_max': value['down_payload_max'],
                'up_payload_std': value['up_payload_std'],
                'down_payload_std': value['down_payload_std'],
                'up_time_interval_mean': value['up_time_interval_mean'],
                'down_time_interval_mean': value['down_time_interval_mean'],
                'up_time_interval_min': value['up_time_interval_min'],
                'down_time_interval_min': value['down_time_interval_min'],
                'up_time_interval_max': value['up_time_interval_max'],
                'down_time_interval_max': value['down_time_interval_max'],
                'up_time_interval_std': value['up_time_interval_std'],
                'down_time_interval_std': value['down_time_interval_std'],
            })


if __name__ == "__main__":
    pcap_file = "PCAP/benign.pcap"
    flow_dict = defaultdict(lambda: {
        'protocol_name': 'Unknown', 'start_time': float('inf'), 'end_time': float('-inf'),
        'up_pkt_count': 0, 'down_pkt_count': 0, 'up_payload_count': 0, 'down_payload_count': 0,
        'up_payload_sizes': [], 'down_payload_sizes': [], 'up_pkt_times': [], 'down_pkt_times': []
    })
    packets = rdpcap(pcap_file)

    for packet in packets:
        process_packet(packet)

    flow_statistics(flow_dict)
    write_to_csv()
