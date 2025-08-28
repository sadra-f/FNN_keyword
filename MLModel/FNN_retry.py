import numpy as np




class FeedForwardNeuralNetwork:
    """
    A Classifier Feed Forward Neural Network Built From Scratch With Numpy.
    """
    def __init__(self):
        self.structure = None
        self.z = []
        self.a = []
        self.err = []
        self.grad = []
        # self.batch_size = 100

    def build(self, inp, hidden:list[list], outp):
        if outp < 2:
            raise ValueError("The NN is designed as to process binary classification as 2 class classification so outp must be >=2")
        self.layout = [inp] + hidden + [outp]
        self.structure = [[f"I{i}" for i in range(self.layout[0])]]
        self._weight_template = []
        for i in range(1, len(self.layout)-1, 1):
            tmp = []
            tmp_w = []
            for j in range(self.layout[i]):
                # +1 in range for the bias
                tmp.append([f"L{i}-N{j}-W{w}" for w in range(self.layout[i-1] + 1)])
                tmp_w.append(np.random.rand(self.layout[i-1] + 1))
            self.structure.append(tmp)
            self._weight_template.append(tmp_w)
        # +1 in range for the bias
        self.structure.append([[f"O{j}-W{w}" for w in range(self.layout[-2] + 1)] for j in range(self.layout[-1])])
        self._weight_template.append([np.random.rand(self.layout[-2] + 1) for j in range(self.layout[-1])])

    def train(self, X, Y):
        self.weights = self._weight_template.copy()
        self.z = []
        self.a = []
        for _, iv in enumerate(self.layout[1:]):
            self.z.append([None for _ in range(iv)])
            self.a.append([None for _ in range(iv)])
        for v in X:
            self._forward(v)
            pass # TODO : add softmax, loss calc and backprop

    def _forward(self, x):
        _x = np.insert(x, 0, 1)
        for j in range(self.layout[1]):
            self.z[0][j] = np.sum(self.weights[0] * _x)
            self.a[0][j] = self._leaky_relu(self.z[0][j])
        for i in range(1, len(self.z)-1, 1):
            for j in range(len(self.z[i])):
                self.z[i][j] = np.sum(self.weights[i][j] * np.insert(self.z[i-1], 0, 1))
                self.a[i][j] = self._leaky_relu(self.z[i][j])
        for j in range(len(self.z[-1])):
            self.z[-1][j] = np.sum(self.weights[-1][j] * np.insert(self.z[-2], 0, 1))
            self.a[-1][j] = self.soft_max(self.z[-1][j])
        return

    def soft_max(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def cross_entropy(y, p):
        return -np.sum(y * np.log(p))


    def _loss(self, Y):
        # the label data is expected as one hot encoded values and even for binary calssification the
        # value should look like [0,1] and not 1
        # and for 4 classes it would be like e.g [0,0,1,0]
        return -(1/len(Y)) * np.sum(Y * np.log(self.a[-1]))
    
    def _onehot_encode(self, y):
        tmp = [-1 for i in range(self.layout[-1])]
        tmp[y] = 1
        return tmp
    
    def _leaky_relu(self, inp):
        const = 0.01
        return np.maximum(inp, (const * inp))