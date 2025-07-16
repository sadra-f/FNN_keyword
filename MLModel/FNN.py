import numpy as np
class FeedForwardNeuralNetwork:
    def __init__(self):
        self.input_features = None
        self.nn = None
        self._ref_nn_zeros = None
        self.batch_size = 1
        self.layer_sizes = None
        self.intermediate_activation = self._sigmoid
        self.output_activation = self._softmax
        self.cost_func = self._classification_cost
        self.add_bias = False
        self.learning_rate = 0.01
        self.cost_history = []
        self.build(3, [5, 5, 3], 2)
        self.train([[1,2,3],[4,5,6],[7,8,9]], [0, 1, 1])

    def build(self, inp_size, hidden_layer_sizes:list, output_size):
        # build the neural netwrok weights in a list excluding the input layer
        self.layer_sizes = list(hidden_layer_sizes)
        self.layer_sizes.insert(0, inp_size)
        self.layer_sizes.append(output_size)
        self.nn = []
        for i in range(1, len(self.layer_sizes), 1):
            # building the layers
            self.nn.append(np.array([np.random.rand(self.layer_sizes[i-1] + (1 if self.add_bias else 0)) for _ in range(self.layer_sizes[i])]))
        return
    

    def train(self, X, Y):
        training_counter = 0
        while training_counter < len(X):
            batch = X[training_counter:training_counter + self.batch_size]
            batch_Y = Y[training_counter:training_counter + self.batch_size]
            self.current_res = np.array([np.zeros(i) for i in self.layer_sizes[1:]], dtype=object)
            b_delta = self.current_res.copy()
            # forward pass
            for vx, vy in zip(batch, batch_Y):
                self._forward_pass(vx)
                Y = np.zeros(len(self.nn[-1]))
                Y[vy] = 1
                self.cost_history.append(self.cost_func(Y))
                l_delta = self._back_propagate(Y)
                b_delta += l_delta # passed error claculation got stuck here FFS... andrew and his broken annotations and incompelete explainations..
                pass
            # self._update_weights(b_delta)
            training_counter = training_counter + self.batch_size
            

    def _forward_pass(self, features):
        # Due to the educational and exploratory nature of this project it is an option to include the bias value
        if self.add_bias:
            self.current_res[0] = self.intermediate_activation(np.sum(np.multiply(self.nn[0], np.insert(features, 0, 1)), axis=1))
            for i in range(1, len(self.nn) - 1, 1):
                self.current_res[i] = self.intermediate_activation(np.sum(np.multiply(self.nn[i], np.insert(self.current_res[i-1], 0, 1)), axis=1))
            self.current_res[-1] = self.output_activation(np.sum(np.multiply(self.nn[-1], np.insert(self.current_res[-2], 0, 1)), axis=1))
        else:
            self.current_res[0] = self.intermediate_activation(np.sum(np.multiply(self.nn[0], features), axis=1))
            for i in range(1, len(self.nn) - 1, 1):
                self.current_res[i] = self.intermediate_activation(np.sum(np.multiply(self.nn[i], self.current_res[i-1]), axis=1))
            self.current_res[-1] = self.output_activation(np.sum(np.multiply(self.nn[-1], self.current_res[-2]), axis=1))

    def _sigmoid(self, inp):
        return 1 / (1 + np.exp(-inp))
    
    def _relu(self, inp):
        return max(inp.max(), 0)
    
    def _leaky_relu(self, inp):
        const = 0.01
        return max(inp.max(), (const * inp).max())
    
    def _softmax(self, inp):
        return np.exp(inp) / np.sum(np.exp(inp), axis=0)
    
    def _classification_cost(self, Y):
        # one hot encode y value to align with classification 

        cost = np.sum(Y * np.log(self.current_res[-1]) + (1 - Y) * np.log(1 - self.current_res[-1]))
        return cost


    def _regression_cost(self):
        pass

    def _back_propagate(self, Y):
        l_delta = self.current_res.copy()
        l_delta[-1] = self.current_res[-1] - Y
        # nn length is subtracted by 2, 1 for len giving the len and no the last index and 1 for
        # the last layer being output layer the value for which is calculated before the loop
        for ri in range(len(self.nn) - 2, 0, -1):
            # TODO : rewatch / continue watching the course there are holes in knowledge as 
            # to how to compute the gradients and how to apply the result of the partial derivatives(b_delta) to each weights in each node of a layer 
            l_delta[ri] = np.matmul(self.nn[ri+1].T, l_delta[ri+1]) * (self.current_res[ri] ** 2)
            # the problem was that the vectors mismatched in size when using the andrew ng explaination somehow its apparently different as implemented above and not this as he said in his vid: np.matmul(self.nn[ri].T, l_delta[ri+1])...

        return l_delta