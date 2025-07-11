import numpy as np
class FeedForwardNeuralNetwork:
    def __init__(self):
        self.input_features = None
        self.nn = None
        self.batch_size = 1
        self.layer_sizes = None
        self.build(3, [5, 5, 3], 2)
        self.train([[1,2,3],[4,5,6],[7,8,9]], None)

    def build(self, inp_size, hidden_layer_sizes:list, output_size):
        # build the neural netwrok weights in a list excluding the input layer
        self.layer_sizes = list(hidden_layer_sizes)
        self.layer_sizes.insert(0, inp_size)
        self.layer_sizes.append(output_size)
        self.nn = []
        for i in range(1, len(self.layer_sizes), 1):
            # building the layers
            self.nn.append([np.random.rand(self.layer_sizes[i-1]) for _ in range(self.layer_sizes[i])])
        return
    

    def train(self, X, Y):
        training_counter = 0
        batch = [X[training_counter:training_counter + self.batch_size]]
        b_delta = np.zeros_like(self.nn)
        # forward pass
        self.current_res = np.array([np.zeros(i) for i in self.layer_sizes[1:]])
        for value in batch:
            pass # continue forward pass

    def _forward_pass(self, features):
        #TODO self explanitory...
