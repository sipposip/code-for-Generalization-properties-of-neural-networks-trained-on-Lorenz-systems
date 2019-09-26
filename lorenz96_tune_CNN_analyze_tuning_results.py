"""


"""



import pickle
import json

import numpy as np
from sklearn.model_selection import ParameterGrid



tunable_params = dict(
                  lr=[0.00001,0.00003,0.0001,0.003],
                  kernel_size = [3,5,7,9],
                  conv_depth = [32,64,128],
                  n_conv=list(range(1,10)),
            activation=['sigmoid', 'relu']
        )


param_grid = list(ParameterGrid(tunable_params))
print(f'trying {len(param_grid)} param combinations')

for lead_time in (1,10,100):
    res = []
    res_params = []
    for i in range(len(param_grid)):

        r = pickle.load(open(f'tune_l95/tunehist_F_leadtime{lead_time}_paramcombi_{i}_.pkl','rb'))
        res.append(np.min(r['hist']['val_loss']))
        res_params.append(r['params'])

    best_idx = np.argmin(res)

    best_params = res_params[best_idx]
    print(best_params)
    #print(param_grid[best_idx])

    json.dump(best_params, open(f'lorenz95_bestCNNparams_leadtime{lead_time}.json', 'w'))


