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

affineList =  [a1, a2, a3,]
affineListBackward = [a3, a2, a1,]


# adaG
# hl = []
# for affine in affineList:
#     hl.append({
#         'a': affine,
#         'hW': np.zeros_like(affine.w),
#         'hb': np.zeros_like(affine.b),
#     })
    
    
# adam
mvl = []
iter = 0
beta1 = 0.9
beta2 = 0.999
for affine in affineList:
    mvl.append({
        'a': affine,
        'mW': np.zeros_like(affine.w),
        'vW': np.zeros_like(affine.w),
        'mb': np.zeros_like(affine.b),
        'vb': np.zeros_like(affine.b),
    })
    


loopCount = 100000
step = 500
lossTotal = None
for count in range(1, loopCount):

    index = np.random.choice(10000, 100) #[random.randint(0, 10000)]
    x = np.array(x_test[index])
    mtLabels = np.array(t_test[index])
    train = np.identity(10)[mtLabels]


    # foward ********
    for s in seq:
        x = s.fw(x)
        
    loss = last.fw(x, train)
    if lossTotal != None:
        lossTotal = (lossTotal + np.average(loss)) * 0.5
    else:
        lossTotal = np.average(loss)


    result = x.argmax(axis = 1)

    # backword **********
    bkSeq = list(seq)
    bkSeq.reverse()

    dout = last.bk(1.0)
    for s in bkSeq:
        dout = s.bk(dout)
        # print(dout)

    # SDG
    # rate = 0.1
    # for a in affineListBackward:
    #     a.w -= (a.dW * rate)
    #     a.b -= (a.db * rate)
        
        
    # # adaG
    # rate = 0.01
    # for h in hl:
    #     a = h['a']
        
    #     h['hW'] += a.dW * a.dW
    #     a.w -= rate * a.dW / (np.sqrt(h['hW']) + 1e-7)
        
    #     h['hb'] += a.db * a.db
    #     a.b -= rate * a.db / (np.sqrt(h['hb']) + 1e-7)
        
    
    # adam
    rate = 0.01
    iter += 1
    rate_t = rate * np.sqrt(1.0 - beta2 ** iter) / (1.0 - beta1 ** iter)
    for mv in mvl:
        a = mv['a']
        
        mv['mW'] += (1 - beta1) * (a.dW - mv['mW'])
        mv['vW'] += (1 - beta2) * (a.dW ** 2 - mv['vW'])
        a.w -= rate_t * mv['mW'] / (np.sqrt(mv['vW'] + 1e-7))
                
        mv['mb'] += (1 - beta1) * (a.db - mv['mb'])
        mv['vb'] += (1 - beta2) * (a.db ** 2 - mv['vb'])
        a.b -= rate_t * mv['mb'] / (np.sqrt(mv['vb'] + 1e-7))
        

    if count % step == 0:
        print('** 学習の進捗: {:.1f}% ***************************'.format((count / loopCount) * 100.0))
        print(' - loss', lossTotal)
        print(' - 正解率', np.sum(result == mtLabels) / mtLabels.shape[0] * 100.0)
        print('')


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





