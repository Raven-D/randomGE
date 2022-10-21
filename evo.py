import numpy as np
import time
import copy
import os

from rlog import _log_warning, _log_info, _log_normal, _log_error

from hparams import params as H

import mnist

random = np.random

def create_weights(shape, mean=0.0, var=1.0):
    return random.normal(mean, var, shape)

def tanh(x):
    return np.tanh(x) * 2.0

def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps, axis=-1).reshape([x.shape[0], 1])

def simple_softmax(x):
    x = x - np.min(x, axis=-1).reshape(x.shape[0], 1)
    return x / np.sum(x, axis=-1).reshape([x.shape[0], 1])


test_data = mnist.test_data
test_label = mnist.test_label

fst = True

class Creature(object):

    def __init__(self, hp, restore_dir='./model', model_fn='evo.npy'):
        self.neuron_count = hp['neuron_count']
        self.layer_count = hp['layer_count']
        self.data_in_dimention = hp['data_in_dimention']
        self.data_out_dimention = hp['data_out_dimention']
        self.erate = hp['erate']
        self.cpu_sleep_step = hp['cpu_sleep_step']
        self.cpu_sleep_time = hp['cpu_sleep_time']
        self.auto_save_step = hp['auto_save_step']
        self.restore_dir = restore_dir
        self.model_fn = model_fn
        self.layers = []
        self.bc_layers = []
        self.bc_ce = np.inf
        self.bacc = 0
        self.step = 1
        # check if has old model.
        self.__restore__()

    def __restore__(self):
        matrix = None
        if (os.path.exists(self.restore_dir)):
            if (os.path.exists(os.path.join(self.restore_dir, self.model_fn))):
                try:
                    matrix = np.load(os.path.join(self.restore_dir, self.model_fn))
                except IOError:
                    _log_error('model file has error...')
        else:
            os.makedirs(self.restore_dir)
        self.__build__model__(matrix)

    def __build__model__(self, matrix):
        if (None != matrix):
            _log_info('restore model from stored model file...')
            self.layers.append(matrix[0])
            if (self.layer_count > 2):
                for i in range(self.layer_count - 2):
                    self.layers.append(matrix[i+1])
            # the index -2 is the last layer, the index -1 is the last bc_ce value.
            self.layers.append(matrix[-4])
            self.bc_ce = matrix[-3]
            self.bacc = matrix[-2]
            self.step = matrix[-1]
        else:
            _log_info('create new model with random weights...')
            self.layers = []
            self.layers.append(create_weights([self.data_in_dimention, self.neuron_count]))
            if (self.layer_count > 2):
                for i in range(self.layer_count - 2):
                    self.layers.append(create_weights([self.neuron_count, self.neuron_count]))
            self.layers.append(create_weights([self.neuron_count, self.data_out_dimention]))

    def __store__(self):
        # SAVE PROTOCOL
        # [weights, bc_ce, bacc, step]
        if (not os.path.exists(self.restore_dir)):
            os.makedirs(self.restore_dir)
        sdata = self.bc_layers
        sdata.append(self.bc_ce)
        sdata.append(self.bacc)
        sdata.append(self.step)
        sdata = np.array(sdata)
        np.save(os.path.join(self.restore_dir, self.model_fn), sdata)
        _log_info('model has been stored, ' + os.path.join(self.restore_dir, self.model_fn))

    def __sleep_cpu__(self):
        if (self.step > 0):
            if (self.step % self.cpu_sleep_step == 0):
                _log_normal('CPU sleep for ' + str(self.cpu_sleep_time) + 's...')
                time.sleep(self.cpu_sleep_time)

    def mutate(self):
        self.bc_layers = copy.deepcopy(self.layers)
        accu_mutate_count = 0
        for i in range(self.layer_count):
            if (random.randint(1, 4) != 1 and i != (self.layer_count - 1)):
                # mutate random for different layers.
                continue
            layer = self.layers[i]
            mcount = random.randint(1, 5)

            if (i == (self.layer_count - 1) and accu_mutate_count == 0 and mcount == 0):
                mcount = 1

            for j in range(mcount):
                position = (random.randint(0, layer.shape[0], []), random.randint(0, layer.shape[1], []))
                layer[position[0], position[1]] = create_weights([])
                # layer[position[0], position[1]] = create_weights([]) * random.uniform(0.0, 1.0)
                # layer[position[0], position[1]] = create_weights([]) * layer[position[0], position[1]]
            accu_mutate_count += mcount
            self.layers[i] = layer
        _log_info('%d neurons mutated.' % accu_mutate_count)
        self.step += 1
        self.__sleep_cpu__()

    def recovery(self):
        if (len(self.bc_layers) == len(self.layers)):
            self.layers = copy.deepcopy(self.bc_layers)
        else:
            _log_warning('bc_layers != layers.')

    def forward(self, data):
        idata = data
        for i in range(self.layer_count):
            idata = tanh(np.dot(idata, self.layers[i]))
        return idata

    def evolve(self, data, label, batch=500):
        assert data.shape[1] == self.layers[0].shape[0]
        assert label.shape[1] == self.layers[-1].shape[1]
        
        global fst
        if (not fst):
            self.mutate()

        start = 0
        ces = []
        correct = 0
        while (start < len(data)):
            end = start + batch
            if (end > len(data)):
                end = len(data)
            cdata = data[start:end]
            clabel = label[start:end]
            logits = self.forward(cdata)
            assert logits.shape[1] == clabel.shape[1]
            probs = softmax(logits)
            ces.append(np.mean(-np.sum(clabel * np.log(probs), axis=-1)))
            correct += np.sum(np.argmax(probs, axis=-1) == np.argmax(clabel, axis=-1))
            start = end

        self.ce = np.mean(ces)
        self.acc = correct / float(len(label)) * 100
        if (self.ce >= self.bc_ce):
            _log_normal('step: %d - mutate failed. bce:%.4f (acc:%.2f%%) [cce:%.4f]' % (self.step, self.bc_ce, self.bacc, self.ce))
            self.recovery()
        else:
            self.bc_ce = self.ce
            self.bacc = self.acc
            _log_warning('step: %d - mutate failed. bce:%.4f (acc:%.2f%%) [cce:%.4f]' % (self.step, self.bc_ce, self.bacc, self.ce))

        if (self.step % self.auto_save_step == 0):
            self.__store__()
        return self.step

def train():
    creature = Creature(H)
    global fst
    try:
        while (True):
            creature.evolve(test_data, test_label)
            if (fst):
                fst = False
    except KeyboardInterrupt:
        creature.__store__()

train()
