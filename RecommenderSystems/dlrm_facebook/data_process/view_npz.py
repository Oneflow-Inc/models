import numpy as np
path='/dataset/criteo_tb/'
fea_count=path+'day_fea_count.npz'
fea_dict_0=path+'day_fea_dict_5.npz'
loaded = np.load(fea_count)
print(loaded['counts'])
loaded = np.load(fea_dict_0)
print(loaded['unique'])

