"""


"""



import pickle
import sys

import numpy as np
from scipy.integrate import odeint
from sklearn.model_selection import ParameterGrid
import keras
import tensorflow as tf
from keras import backend as K



i_batch = int(sys.argv[1])
n_batch = int(sys.argv[2])


# paramters for experiments
N = 40  # number of variables
F = 8
Nsteps = 10000
tstep=0.01
t_arr = np.arange(0, Nsteps) * tstep

# fixed params neural network
n_epoch = 30


def lorenz96(x,t,F):

  # compute state derivatives
  d = np.zeros(N)
  # first the 3 edge cases: i=1,2,N
  d[0] = (x[1] - x[N-2]) * x[N-1] - x[0]
  d[1] = (x[2] - x[N-1]) * x[0]- x[1]
  d[N-1] = (x[0] - x[N-3]) * x[N-2] - x[N-1]
  # then the general case
  #for i in range(2, N-1):
  #    d[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i]
  d[2:N-1] =  (x[2+1:N-1+1] - x[2-2:N-1-2]) * x[2-1:N-1-1] - x[2:N-1]  ## this is equivalent but faster
  # add the forcing term
  d = d + F

  return d


def make_lorenz_run(y0, Nsteps  , F_arr):
    res = np.zeros((Nsteps,N))
    res[0] = y0
    sol = y0
    for i in range(Nsteps-1):  #-1 because we also include the initial state in Nsteps
        F = F_arr[i]
        # print(i)
        # we make only one step, but we gert a 2d array back. 9with only one element)
        # therefore, we extract this element with [1]
        sol = odeint(lorenz96, sol, t=[0,tstep], args=(F,))[1]
        res[i] = sol
    return res

# we make two runs, started with slightly different initial conditions
# one will be the training and one the test run
x_init1 = F*np.ones(N) # initial state (equilibrium)
x_init1[19] += 0.01 # add small perturbation to 20th variable

F_start = 6
F_end = 7

F_arr = F_start * np.ones(len(t_arr)) + (F_end - F_start) / Nsteps / tstep * t_arr

modelrun_train = make_lorenz_run(x_init1, Nsteps, F_arr)



# remove spinpu
modelrun_train = modelrun_train[500:]
F_arr = F_arr[500:]

# for loezn96, we dont have to normalize per variable, because all should have the same
# st and mean anywary, so we compute the total mean,  and the std for each gridpoint and then
# average all std
norm_mean = modelrun_train.mean()
norm_std = modelrun_train.std(axis=0).mean()
modelrun_train = (modelrun_train  - norm_mean) / norm_std


F_arr_normed = (F_arr - F_arr.mean()) / F_arr.std()


# now add F as a second layer (repeating it for every gridpoint)
modelrun_train_with_F = np.stack([modelrun_train ,np.tile(F_arr_normed, (N,1)).T], axis=2)

class PeriodicPadding(keras.layers.Layer):
    def __init__(self, axis, padding, **kwargs):
        """
        layer with periodic padding for specified axis
        tensor: input tensor
        axis: on or multiple axis to pad along
        padding: number of cells to pad
        """

        super(PeriodicPadding, self).__init__(**kwargs)

        if isinstance(axis, int):
            axis = (axis,)
        if isinstance(padding, int):
            padding = (padding,)

        self.axis = axis
        self.padding = padding

    def build(self, input_shape):
        super(PeriodicPadding, self).build(input_shape)

    # in order to be able to load the saved model we need to define
    # get_config
    def get_config(self):
        config = {
            'axis': self.axis,
            'padding': self.padding,

        }
        base_config = super(PeriodicPadding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, input):

        tensor = input
        for ax, p in zip(self.axis, self.padding):
            # create a slice object that selects everything form all axes,
            # except only 0:p for the specified for right, and -p: for left
            ndim = len(tensor.shape)
            ind_right = [slice(-p, None) if i == ax else slice(None) for i in range(ndim)]
            ind_left = [slice(0, p) if i == ax else slice(None) for i in range(ndim)]
            right = tensor[ind_right]
            left = tensor[ind_left]
            middle = tensor
            tensor = tf.concat([right, middle, left], axis=ax)
        return tensor

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        for ax, p in zip(self.axis, self.padding):
            output_shape[ax] += 2 * p
        return tuple(output_shape)

def train_network(X_train, y_train, lr,kernel_size, conv_depth, n_conv, activation):
    """
    :param X_train:
    :param y_train:
    :param kernel_size:
    :param conv_depth:
    :param n_conv: >=1
    :return:
    """

    n_channel = 2
    n_pad = int(np.floor(kernel_size/2))
    layers = [
        PeriodicPadding(axis=1,padding=n_pad,input_shape=(N ,n_channel)),
        keras.layers.Conv1D(kernel_size=kernel_size,filters=conv_depth, activation=activation,
                            padding='valid')]

    for i in range(n_conv-1):
        layers.append(PeriodicPadding(axis=1,padding=n_pad))
        layers.append(keras.layers.Conv1D(kernel_size=kernel_size,filters=conv_depth, activation=activation,
                            padding='valid'))

    layers.append(PeriodicPadding(axis=1, padding=n_pad))
    layers.append(keras.layers.Conv1D(kernel_size=kernel_size, filters=1,  activation='linear', padding='valid'))


    model = keras.Sequential(layers)

    #  we have to add an empty channel dimension for y_train
    y_train = y_train[..., np.newaxis]
    optimizer = keras.optimizers.adam(lr=lr)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    print(model.summary())
    hist = model.fit(X_train, y_train, epochs=n_epoch, verbose=0, validation_split=0.1 ,
                     callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                                              min_delta=0,
                                                              patience=4,

                                                              verbose=1, mode='auto')]
                     )


    return model, hist


tunable_params = dict(
                  lr=[0.00001,0.00003,0.0001,0.003],
                  kernel_size = [3,5,7,9],
                  conv_depth = [32,64,128],
                  n_conv=list(range(1,10)),
            activation=['sigmoid', 'relu']
        )


param_grid = list(ParameterGrid(tunable_params))
n_combis = len(param_grid)
combis_per_batch = int(np.ceil(n_combis / n_batch))


print(f'trying {len(param_grid)} param combinations')


for lead_time in [1,10,100]:
    print(f'lead_time {lead_time}')
    X_train = modelrun_train_with_F[:-lead_time]
    y_train = modelrun_train[lead_time:]

    res = []
    for i,params in enumerate(param_grid):
        if i in range(combis_per_batch * i_batch, combis_per_batch * (i_batch+1)):
            print(f'param combi {i} out of {len(param_grid)}')
            network, hist =  train_network(X_train, y_train, **params)
            res.append({'hist':hist.history, 'params':params})
            pickle.dump({'hist':hist.history, 'params':params},open(f'tunehist_F_leadtime{lead_time}_paramcombi_{i}_.pkl','wb'))
