
import pickle
import json

import numpy as np
res = pickle.load(open('result_tuning_lorenz63.pkl','rb'))

dens_errors = [e['dens_error'] for e in res]
short_errors = [e['abse_shortterm'] for e in res]

min_idx_dens = np.argmin(dens_errors)
min_idx_short = np.argmin(short_errors)

params_dens = res[min_idx_dens]['params']
params_short = res[min_idx_short]['params']

# both yielded the same params

json.dump(params_dens, open('best_params_lorenz63.json','w'))