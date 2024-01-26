import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable


class ConfusionMatrix(object):

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))  # 初始化混淆矩阵，元素都为0
        self.num_classes = num_classes  # 类别数量，本例数据集类别为6
        self.labels = labels  # 类别标签

    def update(self, true, prediction):
        for t, p in zip(true, prediction):  # pred为预测结果，labels为真实标签
            self.matrix[t, p] += 1  # 根据预测结果和真实标签的值统计数量，在混淆矩阵相应位置+1

    def summary(self):  # 计算指标函数
        # calculate accuracy
        sum_tp = 0
        n = np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_tp += self.matrix[i, i]  # 混淆矩阵对角线的元素之和，也就是分类正确的数量
        acc = sum_tp / n  # 总体准确率
        # print("the model accuracy is ", acc)

        # precision, recall, F1 score
        table = PrettyTable()  # 创建一个表格
        table.field_names = ["", "Precision", "Recall", "F1-scores"]
        for i in range(self.num_classes):  # 精确度、召回率、F1分数的计算
            tp = self.matrix[i, i]
            fp = np.sum(self.matrix[:, i]) - tp
            fn = np.sum(self.matrix[i, :]) - tp
            tn = np.sum(self.matrix) - tp - fp - fn

            precision = round(tp / (tp + fp), 3) if tp + fp != 0 else 0.

            recall = round(tp / (tp + fn), 3) if tp + fn != 0 else 0.  # 每一类准确度

            # specificity = round(tn / (tn + fp), 3) if tn + fp != 0 else 0.

            f1 = round((2 * precision * recall) / (precision + recall), 3) if precision + recall != 0 else 0.

            table.add_row([self.labels[i], precision, recall, f1])

        # print(table)
        return str(acc)

    def plot(self, i, j):  # 绘制混淆矩阵
        matrix = self.matrix
        # print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        # plt.title('Confusion Matrix (acc=' + self.summary() + ')')
        plt.title('Confusion Matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        # plt.axis('off')
        # plt.gcf().set_size_inches(512 / 100, 512 / 100)
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.subplots_adjust(top=1, bottom=0, right=0.93, left=0, hspace=0, wspace=0)
        # plt.margins(0, 0)
        plt.savefig("pictures/epoch_%d/matrix_%d.png" % (i, j), dpi=600, bbox_inches='tight')
        plt.clf()
