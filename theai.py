import numpy as np
import pickle
from dataset.mnist import load_mnist

# import random


# ***********************************
#  "ゼロから作る Deep Learning" より
#   https://github.com/oreilly-japan/deep-learning-from-scratch


# ***********************************
#    学習データを更新(無ければ新規作成):
#       create_nn_weight.pyを実行
#


str = input().strip()
inputList = list(map(float, str.split()))


def aa(dots):
    mat = dots.reshape([28, 28])
    for y in mat:
        line = ''
        for x in y:
            if x >= 0.8:
                line += '#'
            elif x >= 0.5:
                line += '+'
            elif x > 0:
                line += '.'
            else:
                line += ' '
        print(line)
        


# ---------------------------
def get_network():
    with open("number_predect_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)

index = np.random.choice(10000, 7) #[random.randint(0, 10000)]
mx = np.array(x_test[index])
mt = np.array(t_test[index])



# for num in mx:
#     aa(num)



# print('正解：', mt)

mt = np.identity(10)[mt]
# print(mt)

# exit(0)

# ---------------------------

d = get_network()

# print(d.keys())

w1 = np.array(d['W1'])
w2 = np.array(d['W2'])
w3 = np.array(d['W3'])


b1 = np.array(d['b1'])
b2 = np.array(d['b2'])
b3 = np.array(d['b3'])


# print(w1.shape, b1.shape)
# print(w2.shape, b2.shape)
# print(w3.shape, b3.shape)
# exit(0)


class Affine:
    def __init__(self, w, b):
        self.b = b
        self.w = w
    
    def fw(self, x):
        return np.dot(x, self.w) + self.b
    
    def bk(self, dout):
        return np.dot(dout, self.w.T)

class ReLU:
    def __init__(self):
        self.mask = None
    
    def fw(self, x):
        self.mask = (x <= 0)
        return np.maximum(0, x)

    def bk(self, dout):
        dout[self.mask] = 0
        return dout
    
    
class SoftMax:
    def __init__(self):
        pass

    def fw(self, x):
        return self.softmax(x)
    
    
    def softmax(self, x):
        max = np.max(x, axis=-1, keepdims=True)
        exp = np.exp(x - max)
        return exp / np.sum(exp, axis=-1, keepdims=True)
    
class SoftMaxWithLoass(SoftMax):
    def __init__(self, train):
        self.t = train
        self.loss = None
        self.y = None
    
    def fw(self, x):
        self.y = super().fw(x)
        self.loss = self.cross_entropy_error(self.y, self.t)
        return self.loss

    def bk(self, dout = 1.0):
        print(self.t.shape[0])
        batch_size = self.t.shape[0]
        return (self.y - self.t) / batch_size


    def cross_entropy_error(self, y, t):
        delta = 1e-7
        return -np.sum(t * np.log(y + delta), axis=1)


# ----------------------------------------
train = mt # np.random.randn(7, 10)


seq = []

# seq.append(
#     Affine(np.array([
#         [1, 2, 3, 4],
#         [-5, -6, -7, -8],
#         [9, 10, 11, 12],
#     ]), np.array([1, 2, 3, 4]))
# )
seq.append(Affine(w1, b1))
seq.append(ReLU())

# seq.append(
#     Affine(np.array([
#         [0, 1, 1, 2],
#         [-1.2, -1.6, -1.7, -0.8],
#         [1, 2, 3, 4],
#         [1, 0, 1, 3],
#     ]), np.array([1, 2, 3, 4]))
# )
seq.append(Affine(w2, b2))
seq.append(ReLU())

# seq.append(
#     Affine(np.array([
#         [0.1, 1, 1, 2],
#         [-5, -6, -7, -8],
#         [1, 1, 1, 1],
#         [1, 2, 3, 4],
#     ]), np.array([-1, -2, -3, -4]))
# )
seq.append(Affine(w3, b3))
seq.append(ReLU())

last = SoftMaxWithLoass(train)


# x = np.array([
#     [10, 0, 0],
#     [1, -2, 30],
#     [100, 0, 30],
# ])
# x = mx # np.random.randn(7, 784)
x = np.array([inputList])

# foward ********
for s in seq:
    x = s.fw(x)
    
# loss = last.fw(x)


# bkSeq = list(seq)
# bkSeq.reverse()


# backword **********
# dout = last.bk(1.0)
# for s in bkSeq:
#     dout = s.bk(dout)


# print('判定：', x.argmax(axis = 1))

result = x.argmax(axis = 1)

print (result[0])
