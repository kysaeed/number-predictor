# ***********************************
#    学習データを更新(無ければ新規作成):


# from matplotlib.pyplot import axis
import numpy as np
import pickle
from dataset.mnist import load_mnist
# import random


print('----------------------------------')


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
    try:
        with open("number_predect_weight.pkl", 'rb') as f:
            network = pickle.load(f)
        return network
    except Exception as e:
        print(e)
        print(" > 新規ファイルに出力します")
        return None
    

def set_network(network):
    with open("number_predect_weight.pkl", 'wb') as f:
        pickle.dump(network, f)



(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)




# for num in mx:
#     aa(num)

# print(mt)

# exit(0)

# ---------------------------
# initial params 

d = get_network()


if d != None:
    # print(d.keys())

    w1 = np.array(d['W1'])
    w2 = np.array(d['W2'])
    w3 = np.array(d['W3'])


    b1 = np.array(d['b1'])
    b2 = np.array(d['b2'])
    b3 = np.array(d['b3'])
    
else:
    w1 = np.random.randn(784, 50) * np.sqrt(1.0 / 784)
    w2 = np.random.randn(50, 100) * np.sqrt(1.0 / 50)
    w3 = np.random.randn(100, 10) * np.sqrt(1.0 / 100)
    
    b1 = np.random.randn(50) * np.sqrt(1.0 / 784)
    b2 = np.random.randn(100) * np.sqrt(1.0 / 50)
    b3 = np.random.randn(10) * np.sqrt(1.0 / 100)

    
    
    # print(d['b1'].shape)

# exit (1)



# print(w1.shape, b1.shape)
# print(w2.shape, b2.shape)
# print(w3.shape, b3.shape)
# exit(0)


class Affine:
    def __init__(self, w, b):
        self.b = b
        self.w = w
        self.x = None
        self.dW = None
        self.db = None
    
    def fw(self, x):
        self.x = x
        return np.dot(x, self.w) + self.b
    
    def bk(self, dout):
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis = 0)
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
        x = x - np.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
        exps = np.exp(x)
        # exps = x
        # print(exps)
        s = np.sum(exps, axis=-1, keepdims=True)        
        soft = exps / s
        # print('soft ****', np.sum(soft))
        return soft
        
    
class SoftMaxWithLoass(SoftMax):
    def __init__(self):
        self.t = None
        self.loss = None
        self.y = None
    
    def fw(self, x, t):
        self.y = super().fw(x)
        self.t = t
        self.loss = self.cross_entropy_error(self.y, t)
        return self.loss

    def bk(self, dout = 1.0):
        # print(self.t.shape[0])
        batch_size = self.t.shape[0]
        return (self.y - self.t) / batch_size


    def cross_entropy_error(self, y, t):
        delta = 1e-7
        return -np.sum(t * np.log(y + delta), axis=1)


# ----------------------------------------


seq = []


a1 = Affine(w1, b1)
seq.append(a1)
seq.append(ReLU())

a2 = Affine(w2, b2)
seq.append(a2)
seq.append(ReLU())

a3 =  Affine(w3, b3)
seq.append(a3)
seq.append(ReLU())

last = SoftMaxWithLoass()

affineListBackward = [a3, a2, a1,]





for count in range(1, 100000):

    index = np.random.choice(10000, 100) #[random.randint(0, 10000)]
    x = np.array(x_test[index])
    mtLabels = np.array(t_test[index])
    train = np.identity(10)[mtLabels]


    # foward ********
    for s in seq:
        x = s.fw(x)
        
    loss = last.fw(x, train)


    result = x.argmax(axis = 1)

    # backword **********
    bkSeq = list(seq)
    bkSeq.reverse()

    dout = last.bk(1.0)
    for s in bkSeq:
        dout = s.bk(dout)
        # print(dout)

    rate = 0.1
    for a in affineListBackward:
        a.w -= (a.dW * rate)
        a.b -= (a.db * rate)
        

    # TODO: 表示を間引いているだけなので、正しい平均値の表示になおす
    if count % 1000 == 0:
        print ('loss', np.average(loss))
        print('正解率', np.sum(result == mtLabels) / mtLabels.shape[0] * 100.0)


uNet = {
    'W1': a1.w.tolist(),
    'W2': a2.w.tolist(),
    'W3': a3.w.tolist(),
    'b1': a1.b.tolist(),
    'b2': a2.b.tolist(),
    'b3': a3.b.tolist(),
}

set_network(uNet)

# print('base: ', np.sum(last.y, axis=1))

# print('正解：', mtLabels)
# print('判定：', result)





