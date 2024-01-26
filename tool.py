import pickle
import dpkt
import random
import numpy as np
from preprocess import categories
from tqdm import tqdm, trange

max_byte_len = 12


def mask(p):
    p.src = b'\x00\x00\x00\x00'
    p.dst = b'\x00\x00\x00\x00'

    if isinstance(p.data, dpkt.tcp.TCP):
        p.data.sport = 0
        p.data.dport = 0
        p.data.seq = 0
        p.data.ack = 0
        p.data.sum = 0

    elif isinstance(p.data, dpkt.udp.UDP):
        p.data.sport = 0
        p.data.dport = 0
        p.data.sum = 0

    return p


def pkt2feature(data, k):
    data_dict = {'train': {}, 'test': {}}

    for c in categories:
        data_dict['train'][c] = []
        data_dict['test'][c] = []
        all_pkts = []
        p_keys = list(data[c].keys())

        for key in p_keys:
            all_pkts = data[c][key]
            print(len(all_pkts))

        random.Random(1024).shuffle(all_pkts)

        for idx in range(len(all_pkts)):
            pkt = mask(all_pkts[idx])
            raw_byte = pkt.pack()

            byte = []
            pos = []
            for x in range(min(len(raw_byte), max_byte_len)):
                byte.append(int(raw_byte[x]))
                pos.append(x)

            byte.extend([0] * (max_byte_len - len(byte)))
            pos.extend([0] * (max_byte_len - len(pos)))

            if idx in range(k * int(len(all_pkts) * 0.1), (k + 1) * int(len(all_pkts) * 0.1)):
                data_dict['test'][c].append((byte, pos))
            else:
                data_dict['train'][c].append((byte, pos))
    return data_dict


def load_epoch_data(dic_name, train='train'):
    pkt_dict = dic_name[train]
    x, y, label = [], [], []

    for c in categories:
        pkts = pkt_dict[c]
        for byte, pos in pkts:
            x.append(byte)
            y.append(pos)
            label.append(categories.index(c))

    return np.array(x), np.array(y), np.array(label)[:, np.newaxis]


if __name__ == '__main__':

    with open('pro_pkts.pkl', 'rb') as f:
        data = pickle.load(f)
    for i in trange(10, mininterval=2, desc='  - (Building fold dataset)   ', leave=False):
        data_dict = pkt2feature(data, i)
        with open('pro_pkts_%d_noip_fold.pkl' % i, 'wb') as f:
            pickle.dump(data_dict, f)
