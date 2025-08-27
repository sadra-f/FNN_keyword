import numpy as np




class FeedForwardNeuralNetwork:
    def __init__(self):
        self.structure = None
        self.a = []
        self.z = []

    def build(self, inp, hidden:list[list], outp):
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

    
