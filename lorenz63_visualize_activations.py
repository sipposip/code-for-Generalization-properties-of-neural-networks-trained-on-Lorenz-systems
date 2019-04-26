
import os

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras import backend as K

import keras

os.system('mkdir -p plots_animation')

beta = 8. / 3.
rho = 28
sigma = 10


def lorenz(X, t=0):
    x, y, z = X
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z

    dXdt = [dxdt, dydt, dzdt]
    return dXdt


def make_lorenz_run(y0, Nsteps, tstep):
    t = np.arange(0, Nsteps) * tstep
    sol = odeint(lorenz, y0, t)
    return sol


Nsteps = 100000
tstep = 0.01
modelrun1 = make_lorenz_run([8, 1, 1], Nsteps, tstep)
# model run has diension (time,variable), where variable is [x,y,z]
# normalize
modelrun1 = (modelrun1 - modelrun1.mean(axis=0)) / modelrun1.std(axis=0)


def lorenz3dplot_lines(data, title="", **kwargs):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    sc = ax.plot(data[:, 0], data[:, 1], data[:, 2], lw=0.5, **kwargs)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title(title)
    return sc


def lorenz3dplot_scatter(data, title="", **kwargs):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    sc = ax.scatter(data[:, 0], data[:, 1], data[:, 2], lw=0.5, **kwargs)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title(title)
    return sc

def compute_3d_density(data):
    extent = (-3, 3.01)
    spacing = 0.3
    bins = np.arange(extent[0], extent[1], spacing)
    hist3d = np.histogramdd(data, bins=[bins, bins, bins], normed=True)

    return hist3d

# %% now train a neural network

hidden_size = 8  # 15 in https://sci-hub.tw/10.1109/CIMSA.2005.1522829
# 8 in https://www.researchgate.net/publication/322809990_Artificial_neural_networks_model_design_of_Lorenz_chaotic_system_for_EEG_pattern_recognition_and_prediction
n_epoch = 100
lead_time = 1  # in steps
dens_thresh = 0.04
param_string = '_'.join([str(e) for e in (Nsteps, tstep, n_epoch, lead_time, dens_thresh)])


def train_network(X_train, y_train):


    attractor_reconstructed = False
    while not attractor_reconstructed:

        model = keras.Sequential([
            keras.layers.Dense(hidden_size, input_shape=(3,), activation='sigmoid'),

            keras.layers.Dense(3, activation='linear')
        ])

        print(model.summary())

        model.compile(optimizer='adam', loss='mean_squared_error')

        hist = model.fit(X_train, y_train, epochs=n_epoch, verbose=2, validation_split=0.1)

        clim = make_network_climrun(model, X_train)
        dens_model, _ = compute_3d_density(X_train)
        dens_network, _ = compute_3d_density(clim)

        rmse_dens = np.sqrt(np.mean((dens_model - dens_network)**2))
        print('rmse_dense', rmse_dens)
        if rmse_dens < dens_thresh:
            attractor_reconstructed = True


    return model

def make_network_climrun(model, test_data):
    # crreate a "climate run" with the network

    N_clim = Nsteps//2
    # as initial state, use a random state from the test data
    y_init_clim = test_data[np.random.randint(len(test_data))]

    network_clim_run = np.zeros((N_clim + 1, 3))
    network_clim_run[0] = y_init_clim
    state = y_init_clim
    state = np.expand_dims(state, axis=0)
    for i in range(N_clim):
        # print(i,N_clim)
        state = model.predict(state)
        network_clim_run[i + 1] = state

    return network_clim_run


def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output,])
    activations = get_activations([X_batch,0])
    return activations

Nsteps = 100000
tstep = 0.01
modelrun1 = make_lorenz_run([8, 1, 1], Nsteps, tstep)
# model run has diension (time,variable), where variable is [x,y,z]
# normalize
modelrun1 = (modelrun1 - modelrun1.mean(axis=0)) / modelrun1.std(axis=0)

# split in first and second half
train_data_full = modelrun1[:int(Nsteps / 2)]
test_data_full = modelrun1[int(Nsteps / 2):]

# X and y data are the same, but shifted by lead-time

X_train = train_data_full[:-lead_time]
y_train = train_data_full[lead_time:]

X_test = test_data_full[:-lead_time]
y_test = test_data_full[lead_time:]

network_full = train_network(X_train, y_train)

# get the activations of the hidden layer for all X_train
activs = get_activations(network_full,0,X_train)[0]




#%% plot distribution af activations for different regions of the phase space

for lims in ([[-3,3],[-3,3],[-3,3]],
        [[1,3],[-1,3],[1,3]],
            [[-3,3],[-3,0],[-3,3]],
             [[-3,0],[-3,3],[-3,3]],  # left wing
             [[0,3],[-3,3],[-3,3]]): # right wing

    idcs = (X_train[:,0] > lims[0][0]) & (X_train[:,0] < lims[0][1])  & \
           (X_train[:,1] > lims[1][0]) & (X_train[:,1] < lims[1][1])  & \
           (X_train[:,2] > lims[2][0]) & (X_train[:,2] < lims[2][1])

    fig = plt.figure(figsize=(14, 7))

    ax1 = plt.subplot(121, projection='3d')
    ax1.set_xlim(-3,3)
    ax1.set_ylim(-3,3)
    ax1.set_zlim(-3,3)
    ax2 = plt.subplot(122)

    sc = ax1.scatter(X_train[idcs, 0], X_train[idcs, 1], X_train[idcs, 2], lw=0.5)
    ax2.set_xlabel('neuron')
    ax2.set_ylabel('activation')
    ax2.set_ylim(0,1)
    ax2.boxplot(activs[idcs])

    plt.savefig(f'plots_animation/region_activation_lims{lims}.png')



#%% now selectively switch off neurons and make predictions


# copy the network
modified_network = keras.models.clone_model(network_full)
modified_network.build((None,3))
modified_network.compile(optimizer='adam', loss='mean_squared_error')
modified_network.set_weights(network_full.get_weights())


weights_orig,bias_orig = network_full.layers[1].get_weights()
# now set weights from hidden neuron 0 to output layer to zero
weights_new = weights_orig.copy()
# weights array has shape (n_hidden,n_output)
weights_new[0,:] = 0
modified_network.layers[1].set_weights([weights_new,bias_orig])

pred_full = network_full.predict(X_test)
abserr_full = np.abs(pred_full - y_test).mean(axis=1)
pred_mod = modified_network.predict(X_test)
abserr_mod = np.abs(pred_mod - y_test).mean(axis=1)




#%%
fig = plt.figure(figsize=(14, 7))
ax1 = plt.subplot(121, projection='3d')
ax2 = plt.subplot(122)

for i in range(0,3000,2):
    print(i)
    ax1.cla()
    ax2.cla()
    sc = ax1.scatter(X_train[np.newaxis,i, 0], X_train[np.newaxis,i, 1], X_train[np.newaxis,i, 2], lw=0.5)
    ax1.set_xlim(-3,3)
    ax1.set_ylim(-3,3)
    ax1.set_zlim(-3,3)

    ax2.plot(activs[i])
    ax2.set_ylim(0,1)
    ax2.set_xlabel('neuron')
    ax2.set_ylabel('activation')
    plt.savefig(f'plots_animation/{i:03d}.png')


os.system('ffmpeg -pattern_type glob -i "plots_animation/????.png"  video_lorenz_activation.mp4')

