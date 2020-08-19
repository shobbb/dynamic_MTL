import pickle as p
from data.fetcher.data_fetcher import BNWFetcher
from model.simple_mtl import SimpleMTL
import tensorflow as tf
import numpy as np
import json
import argparse
import itertools
import sys
import random


def parse_args():
    parser = argparse.ArgumentParser(description='Run Experiment')
    parser.add_argument('--config-file', type=str)

    return parser.parse_args()

class BayesianNetworkWalk:
    def __init__(self, config, sess):
        self.config = config
        self.sess = sess

        self.model = SimpleMTL(self.config['model'])

        fetchers , relationships = self.get_dataset()
        self.train_fetcher, self.valid_fetcher, self.test_fetcher = fetchers
        _, self.valid_relationships, self.test_relationships = relationships

        self.train_step = tf.compat.v1.train.AdamOptimizer(self.config['lr']).minimize(
            self.model.loss, colocate_gradients_with_ops=True
        )
    
    def get_dataset(self):
        with open(self.config['path'], 'rb') as f:
            data = p.load(f)
            train = data['training']
            valid = data['valid']
            test = data['test']

            batch_size = self.config['batch_size']
            fetchers = (
                BNWFetcher(train['inputs'], train['tasks'], train['relationships'], batch_size),
                BNWFetcher(valid['inputs'], valid['tasks'], valid['relationships'], batch_size),
                BNWFetcher(test['inputs'], test['tasks'], test['relationships'], batch_size)
            )

            relationships = (
                train['relationships'], valid['relationships'], test['relationships']
            )
        
        return fetchers, relationships
    
    def train(self, iterations, print_iters=1000):
        total_loss = []
        total_rel_error = []
        for i, data in enumerate(self.train_fetcher.train_loop()):
            inputs, ys, rels = data
            loss, model_rel, _ = self.sess.run(
                (self.model.loss, self.model.relationships, self.train_step),
                feed_dict = {
                    self.model.input: inputs,
                    self.model.ys: ys,
                }
            )
            total_loss.append(loss)
            total_rel_error.append(np.mean(np.square(model_rel - rels)))
            if (i+1) % print_iters == 0:
                print(
                    f'\t{i+1} Training Loss: {np.mean(total_loss)}, Rel Loss {np.mean(total_rel_error)}'
                )
                sys.stdout.flush()
                total_loss = []
                total_rel_error = []
            if i >= iterations:
                break
    
    def evaluate(self, valid='true'):
        loop = self.valid_fetcher.evaluate_loop() if valid else self.test_fetcher.evaluate_loop()
        total_loss = []
        objects = []
        for inputs, ys, rels in loop:
            loss, model_rels, feats_mat = self.sess.run(
                (self.model.loss, self.model.relationships, self.model._t_to_feats_mat),
                feed_dict = {
                    self.model.input: inputs,
                    self.model.ys: ys,
                }
            )
            total_loss.append(loss)
            objects += list(zip(rels, model_rels, feats_mat))
        if valid:
            print(f'Validation Loss: {np.mean(total_loss)}')
            matrices = random.choices(objects, k=3)
            for gt_rel, model_rel, feats_mat in matrices:
                print('-'*100)
                print(np.array(np.split(
                    np.stack([gt_rel.T.flatten(), model_rel.T.flatten()]).T, 
                    self.model.config['num_tasks']
                )))
                print(feats_mat)
    
    def run(self):
        iterations = self.config['iterations']
        valid_iters = self.config['valid_iters']
        print_iters = self.config['print_iters']
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)
        while iterations > 0:
            print(f'Iteration {self.config["iterations"] - iterations}')
            self.train(valid_iters, print_iters=print_iters)
            self.evaluate()
            iterations -= valid_iters
    
if __name__ == '__main__':
    args = parse_args()
    np.set_printoptions(linewidth=500)
    with open(args.config_file, 'rb') as f:
        config = json.load(f)
    with tf.compat.v1.Session() as sess:
        exp = BayesianNetworkWalk(config, sess)
        exp.run()