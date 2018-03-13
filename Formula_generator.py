import numpy as np
import random


def distributive_property(batch_size=300, static_op=False):
    while True:
        parameters = np.random.randint(low=1, high=9, size=(batch_size, 3))
        features = []
        labels = []
        ops = ['+', '-']
        for i in range(parameters.shape[0]):
            if static_op is True:
                features.append("{}*({}+{})".format(*parameters[i]))
                labels.append("{0}*{1}+{0}*{2}".format(*parameters[i]))
            else:
                op = ops[random.randint(0, 1)]
                features.append("{0}*({1}{3}{2})".format(*parameters[i], op))
                labels.append("{0}*{1}{3}{0}*{2}".format(*parameters[i], op))
        yield (features, labels)


# 產生簡單加法算式
# EX: (1+1, 2)
def simpleadd(batch_size=5000, min=1, max=999, padding=True):
    while True:
        # 隨機產生兩個數，介於1~999之間
        parameters = np.random.randint(low=min, high=max, size=(batch_size, 2))

        # 計算 Feature 的最長長度，用於padding
        # EX: 999+999 長度 7
        feature_max_len = len(str(max))*2+1

        # 計算 Label 的最長長度，用於padding
        # EX: 999+999 = 1998 長度 4
        label_max_len = len(str(max*2))

        features = []
        labels = []
        for i in range(batch_size):
            feature = "{}+{}".format(*parameters[i])
            if padding is True:
                feature = feature + " "*(feature_max_len-len(feature))
            features.append(feature)

            label = "{}".format(str(parameters[i][0] + parameters[i][1]))
            if padding is True:
                label = label + " "*(label_max_len-len(label))
            labels.append(label)
        yield (features, labels)
