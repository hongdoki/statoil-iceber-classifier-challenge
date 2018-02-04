import numpy as np

from transform import Compose

class NumpyDataset(object):
    def __init__(self, x, y, num_classes, format='NCHW', transforms=[], weights=None):
        self.x = x
        self.y = y
        self.num_classes = num_classes
        self.format = format
        self.transforms = Compose(transforms)
        self.subsets = self.set_subsets()
        self.weights = self.set_weights(weights)

    def __getitem__(self, index):
        x = np.array(self.x[index])
        y = self.y[index]
        return self.transforms(x), y 

    def __len__(self):
        return len(self.y)

    def set_subsets(self):
        subsets = []
        for label in range(self.num_classes):
            subsets.append([])
        for i in range(len(self.y)):
            label = self.y[i]
            subsets[label].append(i)
        return subsets

    def set_weights(self, weights):
        if weights is None:
            return [ 1 for i in range(self.num_classes) ]
        else:
            assert len(weights) == self.num_classes
            return weights

