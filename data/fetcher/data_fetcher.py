import numpy as np
import pickle as p

class BNWFetcher:
    def __init__(self, data, ys, rels, batch_size):
        self.data = data
        self.ys = ys
        self.rels = rels
        self.batch_size = batch_size
    
    def evaluate_loop(self):
        data = np.pad(self.data, ((0, self.data.shape[0] % self.batch_size), (0, 0)))
        data_split = np.split(data, data.shape[0] / self.batch_size)

        ys = np.pad(self.ys, ((0, self.ys.shape[0] % self.batch_size), (0, 0)))
        y_split = np.split(ys, ys.shape[0] / self.batch_size)

        rels = np.pad(self.rels, ((0, self.rels.shape[0] % self.batch_size), (0, 0), (0, 0)))
        rel_split = np.split(rels, rels.shape[0] / self.batch_size)
        return zip(data_split, y_split, rel_split)

    def randomize_training_batch(self):
        inds = np.arange(self.data.shape[0])
        np.random.shuffle(inds)
        data = np.pad(
            self.data[inds], ((0, self.data.shape[0] % self.batch_size), (0, 0)), 'wrap'
        )
        data_split = np.split(data, data.shape[0] / self.batch_size)

        ys = np.pad(
            self.ys[inds], ((0, self.ys.shape[0] % self.batch_size), (0, 0)), 'wrap'
        )
        y_split = np.split(ys, ys.shape[0] / self.batch_size)

        rels = np.pad(
            self.rels[inds], ((0, self.rels.shape[0] % self.batch_size), (0, 0), (0, 0)), 'wrap'
        )
        rel_split = np.split(rels, rels.shape[0] / self.batch_size)

        return zip(data_split, y_split, rel_split)

    def train_loop(self):
        # Train loop should loop forever, will stop on client side
        while True:
            training_loop = self.randomize_training_batch()

            for data in training_loop:
                yield data
        
