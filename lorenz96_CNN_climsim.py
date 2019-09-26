"""

this script trains a CNN on lorenz95 and then makes a climate simulation with the network
and plots some aspects of the climate
"""



import pickle

import pandas as pd
import numpy as np
from pylab import plt
import seaborn as sns
from scipy.integrate import odeint
from sklearn.model_selection import ParameterGrid
import keras
import tensorflow as tf
from keras import backend as K

# # set maximum numpber of CPUs to use
# config = tf.ConfigProto(intra_op_parallelism_threads=32, inter_op_parallelism_threads=32,
#                         allow_soft_placement=True)
# session = tf.Session(config=config)
# K.set_session(session)

plt.rcParams['savefig.bbox'] = 'tight'
# paramters for experiments
N = 40  # number of variables
F = 8
Nsteps = 10000
tstep=0.01
t_arr = np.arange(0, Nsteps) * tstep

# fixed params neural network
n_epoch = 30

lead_time=10


def lorenz96(x,t):

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

# we make two runs, started with slightly different initial conditions
# one will be the training and one the test run
x_init1 = F*np.ones(N) # initial state (equilibrium)
x_init1[19] += 0.01 # add small perturbation to 20th variable

modelrun_train = odeint(lorenz96,y0=x_init1, t= t_arr)

x_init2 = F*np.ones(N)
x_init2[1] += 0.05 #

modelrun_test = odeint(lorenz96,y0=x_init2, t= t_arr)

# remove spinpu
modelrun_train = modelrun_train[500:]
modelrun_test = modelrun_test[500:]



# for loezn96, we dont have to normalize per variable, because all should have the same
# st and mean anywary, so we compute the total mean,  and the std for each gridpoint and then
# average all std
norm_mean = modelrun_train.mean()
norm_std = modelrun_train.std(axis=0).mean()
modelrun_train = (modelrun_train  - norm_mean) / norm_std

modelrun_test = (modelrun_test - norm_mean) / norm_std

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

    n_channel = 1 # empty
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

    #  we have to add an empty channel dimension
    y_train = y_train[..., np.newaxis]
    X_train = X_train[..., np.newaxis]
    optimizer = keras.optimizers.adam(lr=lr)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    print(model.summary())
    hist = model.fit(X_train, y_train, epochs=n_epoch, verbose=2, validation_split=0.1 ,
                     callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                                              min_delta=0,
                                                              patience=4,

                                                              verbose=1, mode='auto')]
                     )


    return model, hist


params = {100:{"activation": "sigmoid", "conv_depth": 32, "kernel_size": 5, "lr": 0.003, "n_conv": 9},
          10:{"activation": "relu", "conv_depth": 128, "kernel_size": 5, "lr": 0.003, "n_conv": 2},
          1:{"activation": "relu", "conv_depth": 128, "kernel_size": 3, "lr": 0.003, "n_conv": 1}}

X_train = modelrun_train[:-lead_time]
y_train = modelrun_train[lead_time:]

network, hist =  train_network(X_train, y_train, **params[lead_time])

# make climate simulation, starting from first field of the test run

y_init = modelrun_test[0]
N_clim = int(Nsteps // lead_time)
current_state = y_init[np.newaxis,:,np.newaxis]
res = [np.squeeze(current_state)]
for i in range(N_clim):
    current_state = network.predict(current_state)
    res.append(np.squeeze(current_state))


res = np.array(res)

nplot = 100
plt.figure()
clevs = [-3,-2.5,-2,-1.5,-1,-0.5,0.5, 1, 1.5, 2, 2.5, 3]
plt.subplot(121)
plt.contourf(modelrun_test[::lead_time][:nplot], cmap=plt.cm.RdBu, levels=clevs)
plt.title('model')
plt.xlabel('gridpoint')
plt.ylabel('timestep [0.1 timeunits]')
plt.colorbar()
plt.subplot(122)
plt.contourf(res[:nplot], cmap=plt.cm.RdBu, levels=clevs)
plt.title('network climate')
plt.xlabel('gridpoint')
plt.colorbar()
plt.savefig('lorenz95_CNN_clim_evaluation_waveplot.pdf')



plt.figure(figsize=(10,6))
plt.subplot(211)
plt.boxplot(modelrun_test[::lead_time])
plt.title('model')
plt.subplot(212)
plt.boxplot(res)
plt.title('network climate')
plt.xlabel('gridpoint')
plt.savefig('lorenz95_CNN_clim_evaluation_boxplot.pdf')

# plot one single gridpoint
i = 5
plt.figure(figsize=(10,4))
plt.plot(res[:,i], label='network')
plt.plot(modelrun_test[::lead_time,i], label='model')
sns.despine()
plt.legend()
plt.xlabel('timestep')
plt.ylabel('single gridpoint')
plt.savefig('lorenz95_CNN_clim_evaluation_single_gridpoint.pdf')


plt.figure(figsize=(10,6))
plt.subplot(211)
plt.acorr(modelrun_test[::lead_time,i])
plt.title('model')
plt.subplot(212)
plt.acorr(res[:,i])
plt.title('network climate')
plt.xlabel('timelag in timesteps')
plt.ylabel('autocorrelation')
plt.savefig('lorenz95_CNN_clim_evaluation_acorr.pdf')


