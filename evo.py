import numpy as np
import numpy.random as random
import time
import copy
import os

from rlog import _log_warning, _log_info, _log_normal, _log_bg_blue, _log_bg_pp, _log_error

from hparams import params as H

def create_weights(shape):
    return random.normal(0.0, np.e, shape)

def tanh(x):
    return np.tanh(x)

def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps, axis=-1).reshape([x.shape[0], 1])

class Creature(object):

    def __init__(self, hp, restore_dir='./model', model_fn='evo.npy'):
        self.neuron_count = hp['neuron_count']
        self.layer_count = hp['layer_count']
        self.data_in_dimention = hp['data_in_dimention']
        self.data_out_dimention = hp['data_out_dimention']
        self.erate = hp['erate']
        self.FIX_ERATE = hp['erate']
        self.decay_step = hp['decay_step']
        self.decay_rate = hp['decay_rate']
        self.cpu_sleep_step = hp['cpu_sleep_step']
        self.cpu_sleep_time = hp['cpu_sleep_time']
        self.restore_dir = restore_dir
        self.model_fn = model_fn
        self.layers = []
        self.bc_layers = []
        self.bc_ce = np.inf
        self.step = 0
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
            _log_bg_blue('restore model from stored model file...')
            self.layers.append(matrix[0])
            if (self.layer_count > 2):
                for i in range(self.layer_count - 2):
                    self.layers.append(matrix[i+1])
            # the index -2 is the last layer, the index -1 is the last bc_ce value.
            self.layers.append(matrix[-3])
            self.bc_ce = matrix[-2]
            self.step = matrix[-1]
            self.erate = self.FIX_ERATE * (self.decay_rate ** int(self.step / self.decay_step))
        else:
            _log_bg_blue('create new model with random weights...')
            self.layers = []
            self.layers.append(create_weights([self.data_in_dimention, self.neuron_count]))
            if (self.layer_count > 2):
                for i in range(self.layer_count - 2):
                    self.layers.append(create_weights([self.neuron_count, self.neuron_count]))
            self.layers.append(create_weights([self.neuron_count, self.data_out_dimention]))

    def __store__(self):
        # SAVE PROTOCOL
        # [weights, bc_ce, step]
        if (not os.path.exists(self.restore_dir)):
            os.makedirs(self.restore_dir)
        sdata = self.layers
        sdata.append(self.bc_ce)
        sdata.append(self.step)
        sdata = np.array(sdata)
        np.save(os.path.join(self.restore_dir, self.model_fn), sdata)
        _log_bg_blue('model has been stored, ' + os.path.join(self.restore_dir, self.model_fn))
        

    def __calc_erate__(self):
        if (self.step > 0):
            if (self.step % self.decay_step == 0):
                times = int(self.step / self.decay_step)
                self.erate = self.FIX_ERATE * (self.decay_rate ** times)

    def __sleep_cpu__(self):
        if (self.step > 0):
            if (self.step % self.cpu_sleep_step == 0):
                _log_info('CPU sleep for ' + str(self.cpu_sleep_time) + 's...')
                time.sleep(self.cpu_sleep_time)

    def mutate(self):
        self.bc_layers = copy.deepcopy(self.layers)
        self.__calc_erate__()
        for i in range(self.layer_count):
            layer = self.layers[i]
            count = layer.shape[0] * layer.shape[1]
            mcount = int(count * self.erate)
            if (mcount < 1):
                mcount = 1
            for j in range(mcount):
                position = (random.randint(0, layer.shape[0], []), random.randint(0, layer.shape[1], []))
                layer[position[0], position[1]] = create_weights([])
            self.layers[i] = layer
        self.step += 1
        self.__sleep_cpu__()
                
    def recovery(self):
        self.layers = self.bc_layers

    def forward(self, data):
        idata = data
        for i in range(self.layer_count):
            idata = tanh(np.dot(idata, self.layers[i]))
        return idata        

    def evolve(self, data, target):
        assert data.shape[1] == self.layers[0].shape[0]
        assert target.shape[1] == self.layers[-1].shape[1]

        self.mutate()
        idata = self.forward(data)

        assert idata.shape[1] == target.shape[1]

        idata = softmax(idata)
        self.ce = np.mean(-np.sum(target * np.log(idata), axis=-1))

        if (self.ce >= self.bc_ce):
            self.recovery()
        else:
            self.bc_ce = self.ce
        _log_normal('BCE:%.10f (CE:%f) ERATE:%.4f, STEP:%d' % (self.bc_ce, self.ce, self.erate, self.step))
        return self.step

    def infer(self, data):
        idata = self.forward(data)
        idata = softmax(idata)
        y = np.argmax(idata, -1)
        return y

import mnist

def infer(creature):
    tcount = mnist.get_infer_batch_count(H['infer_batch_size'])
    correction = 0
    while(tcount > 0):
        tex, tey = mnist.get_batch_for_infer(H['infer_batch_size'])
        y = creature.infer(tex)
        tey = np.argmax(tey, -1)
        correction += np.sum(np.equal(y, tey))
        tcount -= 1
    _log_bg_pp('Correction:%.2f%%' % (float(correction) / mnist.TEST_DATA_SIZE * 100))

def train():
    # neuron_count, layer_count, data_in_dimention, data_out_dimention, erate
    creature = Creature(H)
    tindex = 0
    try:
        while (tindex < mnist.TRAIN_DATA_SIZE * 3):
            x, y = mnist.get_batch_for_train(H['train_batch_size'])
            creature.evolve(x, y)
            tindex += 1
            if (tindex % 1000 == 0):
                time.sleep(3)
        # train finish
        infer(creature)
    except KeyboardInterrupt:
        creature.__store__()

if __name__ == '__main__':
    train()
    
