import numpy as np
class FeedForwardNeuralNetwork:
    def __init__(self):
        self.input_features = None
        self.nn = None


    def build(self, inp_size, hidden_layer_sizes:list, output_size):
        sizes = list(hidden_layer_sizes)
        sizes.insert(0, inp_size)
        sizes.append(output_size)
        self.nn = []
        for i in range(1, len(sizes), 1):
            # building the layers
            self.nn.append(np.array([np.random.rand(sizes[i-1]) for i in range(sizes[i])]))