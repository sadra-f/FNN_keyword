import numpy as np
import copy



class FeedForwardNeuralNetwork:
    """
    A Classifier Feed Forward Neural Network Built From Scratch With Numpy.
    """
    def __init__(self):
        self.structure = None
        self._z_template = []
        self._a_template = []
        self._a_template = []
        
        self.z = None
        self.a = None
        self._errors = None
        
        self.gradients = None
        
        self.loss_hist = None
        
        self.RELU_CONST = 0.01
        self.LEARNING_RATE = 0.001
        # self.batch_size = 100

    def build(self, inp, hidden:list[list], outp):
        if outp < 2:
            raise ValueError("The NN is designed as to process binary classification as 2 class classification so outp must be >=2")
        # first, input layer
        self.layout = [inp] + hidden + [outp]
        self.structure = [[f"I{i}" for i in range(self.layout[0])]]
        self._weight_template = []
        # hidden layers
        for i in range(1, len(self.layout)-1, 1):
            tmp = []
            tmp_w = []
            # layer nodes
            for j in range(self.layout[i]):
                # +1 in range for the bias
                tmp.append([f"L{i}-N{j}-W{w}" for w in range(self.layout[i-1] + 1)])
                std_dev = np.sqrt(2.0 / (self.layout[i-1] + 1))
                # node weights
                tmp_w.append(np.random.normal(loc=0.0, scale=std_dev, size=self.layout[i-1]+1))
            self.structure.append(tmp)
            self._weight_template.append(tmp_w)
        #final, output layer
        # +1 in range for the bias
        self.structure.append([[f"O{j}-W{w}" for w in range(self.layout[-2] + 1)] for j in range(self.layout[-1])])
        std_dev = np.sqrt(2.0 / (self.layout[-1] + 1))
        self._weight_template.append(np.random.normal(loc=0.0, scale=std_dev, size=(self.layout[-1], self.layout[-2]+1)))
        # build templates for other parameters
        self._z_template = []
        self._a_template = []
        self._errors_template = []
        for _, iv in enumerate(self.layout[1:]):
            self._z_template.append([None for _ in range(iv)])
            self._a_template.append([None for _ in range(iv)])
            self._errors_template.append([None for _ in range(iv)])

    def train(self, X, Y, epoch=1):
        #initialize parameters
        self.weights = copy.deepcopy(self._weight_template)

        self.loss_hist = []
        
        self.z = copy.deepcopy(self._z_template)
        self.a = copy.deepcopy(self._a_template)
        self._errors = copy.deepcopy(self._errors_template)

        self.gradients = copy.deepcopy(self._weight_template)
        # train the model
        for epc in range(epoch):
            self.loss_hist.append([])
            for i, _x in enumerate(X):
                y = self._onehot_encode(Y[i])
                x = self._normalize(_x)

                self._forward(x)

                self.loss_hist[epc].append(self.cross_entropy(y, self.a[-1]))

                # backpropagation
                self._err(y)
                self._clac_gradient(x)

                self._update_weights()

    def _forward(self, x):
        _x = np.insert(x, 0, 1)
        #for the first layer (separated to simplify implemntation)
        for j in range(self.layout[1]):
            self.z[0][j] = np.sum(self.weights[0][j] * _x)
            self.a[0][j] = self._leaky_relu(self.z[0][j])
        # for the second to L-1 layer
        for i in range(1, len(self.z)-1, 1):
            for j in range(len(self.z[i])):
                self.z[i][j] = np.sum(self.weights[i][j] * np.insert(self.a[i-1], 0, 1))
            self.a[i] = self._leaky_relu(self.z[i])
        # for the last layer
        for j in range(len(self.z[-1])):
            self.z[-1][j] = np.sum(self.weights[-1][j] * np.insert(self.a[-2], 0, 1))
        self.a[-1] = self.soft_max(self.z[-1])
        return

    def predict(self, x, logits=False):
        _x = np.insert(self._normalize(x), 0, 1)
        # make a deepcopy and overwrite since they have the same structure for training and testing
        tmp_z = copy.deepcopy(self.z)
        tmp_a = copy.deepcopy(self.a)
        # first hiddel layer
        for j in range(self.layout[1]):
            tmp_z[0][j] = np.sum(self.weights[0][j] * _x)
            tmp_a[0][j] = self._leaky_relu(tmp_z[0][j])
        # rest of hidden layers
        for i in range(1, len(tmp_z)-1, 1):
            for j in range(len(tmp_z[i])):
                tmp_z[i][j] = np.sum(self.weights[i][j] * np.insert(tmp_a[i-1], 0, 1))
            tmp_a[i] = self._leaky_relu(tmp_z[i])
        # final, output layer
        for j in range(len(tmp_z[-1])):
            tmp_z[-1][j] = np.sum(self.weights[-1][j] * np.insert(tmp_a[-2], 0, 1))
        tmp_a[-1] = self.soft_max(tmp_z[-1])

        if logits:
            return (np.array(tmp_a[-1]).argmax(), tmp_a)
        
        return np.array(tmp_a[-1]).argmax()

    def predict_all(self, X, logits=False):
        res = []
        for v in X:
            res.append(self.predict(X, logits))
        return res

    def soft_max(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def cross_entropy(self, y, p):
        p = np.clip(p, 1e-12, 1.0 - 1e-12)
        return -np.sum(y * np.where(p > 0, np.log(p), 0))

    def _err(self, y):
        self._errors[-1] = self.a[-1] - y
        for i in range(len(self.weights)-2, -1, -1):
            _deriv = np.array(self.z[i])
            _deriv[_deriv > 0] = 1
            _deriv[_deriv <= 0] = self.RELU_CONST
            self._errors[i] = np.matmul(np.array(self.weights[i+1])[:, 1:].T, self._errors[i+1]) * _deriv
        return

    def _clac_gradient(self, inp):
        self.gradients[0] = np.outer(self._errors[0], np.array(inp).T)
        for i in range(1, len(self.weights), 1):
            self.gradients[i] = np.outer(self._errors[i], np.array(self.a[i-1]))
        return

    def _update_weights(self):
        for i, vi in enumerate(self.gradients):
            self.weights[i] = np.array(self.weights[i])
            self.weights[i][:, 0] = self.weights[i][:, 0] - self.LEARNING_RATE * self._errors[i]
            for j, vj in enumerate(vi):
                self.weights[i][j][1:] = self.weights[i][j][1:] - self.LEARNING_RATE * vj
                pass
        return

    def _loss(self, Y):
        # the label data is expected as one hot encoded values and even for binary calssification the
        # value should look like [0,1] and not 1
        # and for 4 classes it would be like e.g [0,0,1,0]
        return -(1/len(Y)) * np.sum(Y * np.log(self.a[-1]))
    
    def _onehot_encode(self, y):
        tmp = [0 for _ in range(self.layout[-1])]
        tmp[y] = 1
        return tmp
    
    def _leaky_relu(self, inp):
        return np.maximum(inp, (self.RELU_CONST * np.array(inp)))
    
    def _normalize(self, vec):
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec  # Return original if zero vector
        return vec / norm