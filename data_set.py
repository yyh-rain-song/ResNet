from copy import copy

import numpy as np


class Data_set(object):

    def __init__(self, train_percent):
        self._global_pointer = 0
        self._near = 48
        self._distant = 48 * 7
        self._data = None
        self._avaliable = 0
        self._train_percent = train_percent
        self._extern = None

    def build(self, input_data, extern_input):
        # _data: [None, 2, 32, 32]
        self._data = input_data.copy()
        self._extern = extern_input.copy()
        self._avaliable = (int(input_data.shape[0]*self._train_percent/48))*48
        self._global_pointer = self._distant + 4

    def take_entry(self, training=True):
        if training:
            self._global_pointer = self._global_pointer % self._avaliable
            if self._global_pointer < self._distant+4:
                self._global_pointer = self._distant+4

        glb = self._global_pointer

        extern = self._extern[glb]
        y = self._data[glb]

        x_recent = self._data[glb-4:glb]
        x_recent = np.vstack((x_recent[0], x_recent[1], x_recent[2], x_recent[3]))
        x_near = self._data[glb-4-self._near: glb-self._near]
        x_near = np.vstack((x_near[0], x_near[1], x_near[2], x_near[3]))
        x_distant = self._data[glb-4-self._distant: glb-self._distant]
        x_distant = np.vstack((x_distant[0], x_distant[1], x_distant[2], x_distant[3]))
        return x_recent, x_near, x_distant, extern, y

    # return x: [batch_size, 3, 4*2, 32, 32] y: [batch_size, 2, 32, 32] extern: [batch_size, 9]
    def next_batch(self, batch_size):
        x_recent = []
        x_near = []
        x_distant = []
        extern = []
        y = []

        for i in range(batch_size):
            x1, x2, x3, e, o = self.take_entry()
            self._global_pointer += 1
            x_recent.append(x1)
            x_near.append(x2)
            x_distant.append(x3)
            extern.append(e)
            y.append(o)

        x = []
        x.append(x_recent)
        x.append(x_near)
        x.append(x_distant)
        x = np.asarray(x)
        y = np.asarray(y)
        extern = np.asarray(extern)
        x = np.swapaxes(x, 1, 0)
        return x, y, extern

    def test_data(self):
        self._global_pointer = self._avaliable
        x_recent = []
        x_near = []
        x_distant = []
        extern = []
        y = []
        for i in range(self._avaliable, self._data.shape[0]):
            x1, x2, x3, e, o = self.take_entry(training=False)
            self._global_pointer += 1
            x_recent.append(x1)
            x_near.append(x2)
            x_distant.append(x3)
            extern.append(e)
            y.append(o)
        x = []
        x.append(x_recent)
        x.append(x_near)
        x.append(x_distant)
        x = np.asarray(x)
        y = np.asarray(y)
        extern = np.asarray(extern)
        x = np.swapaxes(x, 1, 0)
        return x, y, extern
