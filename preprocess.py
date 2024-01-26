import pickle
import dpkt

categories = ["Chat", "Email", "FileTransfer", "Game", "Meeting", "P2P", "Streaming", "Web"]
labels = [0, 1, 2, 3, 4, 5, 6, 7]


def gen_pkts(pcap, label):
    pkts = {}

    if pcap.datalink() != dpkt.pcap.DLT_EN10MB:
        print('unknown data link!')
        return

    pkts[str(label)] = []

    for _, buff in pcap:

        eth = dpkt.ethernet.Ethernet(buff)

        if isinstance(eth.data, dpkt.ip.IP) and (
                isinstance(eth.data.data, dpkt.udp.UDP)
                or isinstance(eth.data.data, dpkt.tcp.TCP)):
            ip = eth.data
            pkts[str(label)].append(ip)

    return pkts


def closure(pkts):

    pkts_dict = {}
    for name in categories:
        index = categories.index(name)
        pkts_dict[name] = pkts[index]
        cnt = 0
        for k, v in pkts[index].items():
            cnt += len(v)
        print('============================')
        print('Generate pkts for %s' % name)
        print('Total pkts: ', cnt)

    with open('pro_pkts.pkl', 'wb') as f:
        pickle.dump(pkts_dict, f)


if __name__ == '__main__':
    pkts_list = []

    chat = dpkt.pcap.Reader(open('Dataset/Chat.pcap', 'rb'))
    chat_dic = gen_pkts(chat, 0)
    pkts_list.append(chat_dic)

    email = dpkt.pcap.Reader(open('Dataset/Email.pcap', 'rb'))
    email_dic = gen_pkts(email, 1)
    pkts_list.append(email_dic)

    ft = dpkt.pcap.Reader(open('Dataset/FileTransfer.pcap', 'rb'))
    ft_dic = gen_pkts(ft, 2)
    pkts_list.append(ft_dic)

    game = dpkt.pcap.Reader(open('Dataset/Game.pcap', 'rb'))
    game_dic = gen_pkts(game, 3)
    pkts_list.append(game_dic)

    meeting = dpkt.pcap.Reader(open('Dataset/Meeting.pcap', 'rb'))
    meeting_dic = gen_pkts(meeting, 4)
    pkts_list.append(meeting_dic)

    p2p = dpkt.pcap.Reader(open('Dataset/P2P.pcap', 'rb'))
    p2p_dic = gen_pkts(p2p, 5)
    pkts_list.append(p2p_dic)

    streaming = dpkt.pcap.Reader(open('Dataset/Streaming.pcap', 'rb'))
    streaming_dic = gen_pkts(streaming, 6)
    pkts_list.append(streaming_dic)

    web = dpkt.pcap.Reader(open('Dataset/Web.pcap', 'rb'))
    web_dic = gen_pkts(web, 7)
    pkts_list.append(web_dic)

    closure(pkts_list)
