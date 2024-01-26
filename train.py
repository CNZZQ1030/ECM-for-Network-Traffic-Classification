import os
import torch
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
import time
from tqdm import tqdm, trange
from evaluation import ConfusionMatrix
import configure
from model import ECM
from tool import categories, load_epoch_data, max_byte_len
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import pickle


class Dataset(torch.utils.data.Dataset):
    """docstring for Dataset"""

    def __init__(self, x, y, label):
        super(Dataset, self).__init__()
        self.x = x
        self.y = y
        self.label = label

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.label[idx]


def paired_collate_fn(insts):
    x, y, label = list(zip(*insts))
    return torch.LongTensor(x), torch.LongTensor(y), torch.LongTensor(label)


def cal_loss(pred, gold, cls_ratio=None):
    gold = gold.contiguous().view(-1)
    loss = F.cross_entropy(pred, gold)
    pred = F.softmax(pred, dim=-1).max(1)[1]
    n_correct = pred.eq(gold)
    acc = n_correct.sum().item() / n_correct.shape[0]

    return loss, acc * 100


def test_epoch(model, test_data):
    """ Epoch operation in training phase"""
    model.eval()

    total_acc = []
    total_pred = []
    total_time = []
    with torch.no_grad():
        for batch in tqdm(
                test_data, mininterval=2,
                desc='  - (Testing)   ', leave=False):
            # prepare data
            src_seq, src_seq2, gold = batch
            src_seq, src_seq2, gold = src_seq.cuda(), src_seq2.cuda(), gold.cuda()
            gold = gold.contiguous().view(-1)

            # forward
            torch.cuda.synchronize()
            start = time.time()
            pred = model(src_seq, src_seq2)
            torch.cuda.synchronize()
            end = time.time()
            n_correct = pred.eq(gold)

            acc = n_correct.sum().item() * 100 / n_correct.shape[0]
            total_acc.append(acc)
            total_pred.extend(pred.long().tolist())
            total_time.append(end - start)

    return sum(total_acc) / len(total_acc), total_pred, sum(total_time) / len(total_time)


def train_epoch(model, training_data, optimizer):
    """ Epoch operation in training phase"""
    model.train()

    total_loss = []
    total_acc = []

    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):
        src_seq, src_seq2, gold = batch
        src_seq, src_seq2, gold = src_seq.cuda(), src_seq2.cuda(), gold.cuda()

        optimizer.zero_grad()
        pred = model(src_seq, src_seq2)
        loss_per_batch, acc_per_batch = cal_loss(pred, gold)
        loss_per_batch.backward()
        optimizer.step()

        total_loss.append(loss_per_batch.item())
        total_acc.append(acc_per_batch)

    return sum(total_loss) / len(total_loss), sum(total_acc) / len(total_acc)


def main(i, dic_name):
    f1 = open('results/results_%d.txt' % i, 'w')
    f1.write('Train Loss Time Test\n')
    f1.flush()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    model = ECM(num_class=8, max_byte_len=12).cuda()
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()))
    loss_list = []
    for epoch_i in trange(200, mininterval=2,
                          desc='  - (Training Epochs)   ', leave=False):
        class_indict = configure.traffic_DICT
        label = [label for _, label in class_indict.items()]
        confusion = ConfusionMatrix(num_classes=configure.NUM_CLASSES, labels=label)

        train_x, train_y, train_label = load_epoch_data(dic_name, 'train')
        training_data = torch.utils.data.DataLoader(
            Dataset(x=train_x, y=train_y, label=train_label),
            num_workers=0,
            collate_fn=paired_collate_fn,
            batch_size=128,
            shuffle=True
        )
        train_loss, train_acc = train_epoch(model, training_data, optimizer)

        test_x, test_y, test_label = load_epoch_data(dic_name, 'test')
        test_data = torch.utils.data.DataLoader(
            Dataset(x=test_x, y=test_y, label=test_label),
            num_workers=0,
            collate_fn=paired_collate_fn,
            batch_size=1,
            shuffle=False
        )
        test_acc, pred, test_time = test_epoch(model, test_data)

        with open('results/metric_%d.txt' % i, 'w') as f3:
            f3.write('F1 PRE REC\n')
            p, r, fscore, _ = precision_recall_fscore_support(test_label, pred)
            for a, b, c in zip(fscore, p, r):
                f3.write('%.2f %.2f %.2f\n' % (a, b, c))
                f3.flush()
            if len(fscore) != len(categories):
                a = set(pred)
                b = set(test_label[:, 0])
                f3.write('%s\n%s' % (str(a), str(b)))

        f1.write('%.2f %.4f %.6f %.2f\n' % (train_acc, train_loss, test_time, test_acc))
        f1.flush()

    f1.close()


if __name__ == '__main__':
    for i in range(10):
        with open('pro_pkts_%d_noip_fold.pkl' % i, 'rb') as f:
            data_dict = pickle.load(f)
        print('====', i, ' fold validation ====')
        main(i, data_dict)
