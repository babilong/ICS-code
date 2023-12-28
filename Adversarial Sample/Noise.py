import random
import numpy as np
import pandas as pd
import scipy.io as scio
import math

data_TT = pd.read_csv('data_ICS_6_TT_multi_train_shuffle_zscore.csv')
# data_TT = pd.read_csv('data_ICS_6_TT_multi_test_shuffle_zscore.csv')

data = data_TT.copy()
data0 = data_TT.copy()
data_TTT = data_TT.copy()

data_TTT.to_csv('dataset/multi/data_ICS_6_TT_multi_noise_0.csv', index=False)
data_TTT.iloc[8000:32000, 27:28] = 1
data_TTT.to_csv('dataset/binary/data_ICS_6_TT_binary_noise_0.csv', index=False)
print('Raw All Down')

noise_df_gauss = pd.read_excel('Noise/noise_Gauss.xlsx')
noise_np_gauss = np.array(noise_df_gauss)
nn0_gauss = []
for i in range(0, 10000):
    nn0_gauss.append(noise_np_gauss[i][1] / 20)

noise_df = pd.read_excel('Noise/noise_Allcolor.xlsx')
noise_np = np.array(noise_df)
nn0 = []
for i in range(0, 10000):
    nn0.append(noise_np[i][1] / 20)

nn0_gauss = np.random.choice(nn0_gauss, 8000)
nn0 = np.random.choice(nn0, 8000)

for i in range(0, 8000):
    if nn0_gauss[i] < 0:
        nn0_gauss[i] = -nn0_gauss[i]
    if nn0[i] < 0:
        nn0[i] = -nn0[i]


def get_noisy_multi_0(_data):
    for kk in range(1, 21):
        for i in range(0, 8000):
            _data.loc[i, 'duration'] += nn0[i]
            if _data.iloc[i, 0] != 0:
                _data.loc[i, 'up_time_interval_mean'] += nn0[i]
                _data.loc[i, 'up_time_interval_min'] += nn0[i]
                _data.loc[i, 'up_time_interval_max'] += nn0[i]
                _data.loc[i, 'up_time_interval_std'] += nn0[i]

                _data.loc[i, 'up_payload_mean'] += nn0_gauss[i]
                _data.loc[i, 'up_payload_std'] += nn0_gauss[i]

                _data.loc[i, 'up_pkt_count'] += nn0_gauss[i]
                _data.loc[i, 'up_payload_count'] += nn0_gauss[i]
                _data.loc[i, 'up_payload_min'] += nn0_gauss[i]
                _data.loc[i, 'up_payload_max'] += nn0_gauss[i]
            else:
                _data.loc[i, 'down_time_interval_mean'] += nn0[i]
                _data.loc[i, 'down_time_interval_min'] += nn0[i]
                _data.loc[i, 'down_time_interval_max'] += nn0[i]
                _data.loc[i, 'down_time_interval_std'] += nn0[i]

                _data.loc[i, 'down_payload_mean'] += nn0_gauss[i]
                _data.loc[i, 'down_payload_std'] += nn0_gauss[i]

                _data.loc[i, 'down_pkt_count'] += nn0_gauss[i]
                _data.loc[i, 'down_payload_count'] += nn0_gauss[i]
                _data.loc[i, 'down_payload_min'] += nn0_gauss[i]
                _data.loc[i, 'down_payload_max'] += nn0_gauss[i]
        for i in range(8000, 16000):
            _data.loc[i, 'duration'] += nn0[i - 8000]
            if _data.iloc[i, 0] != 0:
                _data.loc[i, 'up_time_interval_mean'] += nn0[i - 8000]
                _data.loc[i, 'up_time_interval_min'] += nn0[i - 8000]
                _data.loc[i, 'up_time_interval_max'] += nn0[i - 8000]
                _data.loc[i, 'up_time_interval_std'] += nn0[i - 8000]

                _data.loc[i, 'up_payload_mean'] += nn0_gauss[i - 8000]
                _data.loc[i, 'up_payload_std'] += nn0_gauss[i - 8000]

                _data.loc[i, 'up_pkt_count'] += nn0_gauss[i - 8000]
                _data.loc[i, 'up_payload_count'] += nn0_gauss[i - 8000]
                _data.loc[i, 'up_payload_min'] += nn0_gauss[i - 8000]
                _data.loc[i, 'up_payload_max'] += nn0_gauss[i - 8000]
            else:
                _data.loc[i, 'down_time_interval_mean'] += nn0[i - 8000]
                _data.loc[i, 'down_time_interval_min'] += nn0[i - 8000]
                _data.loc[i, 'down_time_interval_max'] += nn0[i - 8000]
                _data.loc[i, 'down_time_interval_std'] += nn0[i - 8000]

                _data.loc[i, 'down_payload_mean'] += nn0_gauss[i - 8000]
                _data.loc[i, 'down_payload_std'] += nn0_gauss[i - 8000]

                _data.loc[i, 'down_pkt_count'] += nn0_gauss[i - 8000]
                _data.loc[i, 'down_payload_count'] += nn0_gauss[i - 8000]
                _data.loc[i, 'down_payload_min'] += nn0_gauss[i - 8000]
                _data.loc[i, 'down_payload_max'] += nn0_gauss[i - 8000]
        for i in range(16000, 24000):
            _data.loc[i, 'duration'] += nn0[i - 16000]
            if _data.iloc[i, 0] != 0:
                _data.loc[i, 'up_time_interval_mean'] += nn0[i - 16000]
                _data.loc[i, 'up_time_interval_min'] += nn0[i - 16000]
                _data.loc[i, 'up_time_interval_max'] += nn0[i - 16000]
                _data.loc[i, 'up_time_interval_std'] += nn0[i - 16000]

                _data.loc[i, 'up_payload_mean'] += nn0_gauss[i - 16000]
                _data.loc[i, 'up_payload_std'] += nn0_gauss[i - 16000]

                _data.loc[i, 'up_pkt_count'] += nn0_gauss[i - 16000]
                _data.loc[i, 'up_payload_count'] += nn0_gauss[i - 16000]
                _data.loc[i, 'up_payload_min'] += nn0_gauss[i - 16000]
                _data.loc[i, 'up_payload_max'] += nn0_gauss[i - 16000]
            else:
                _data.loc[i, 'down_time_interval_mean'] += nn0[i - 16000]
                _data.loc[i, 'down_time_interval_min'] += nn0[i - 16000]
                _data.loc[i, 'down_time_interval_max'] += nn0[i - 16000]
                _data.loc[i, 'down_time_interval_std'] += nn0[i - 16000]

                _data.loc[i, 'down_payload_mean'] += nn0_gauss[i - 16000]
                _data.loc[i, 'down_payload_std'] += nn0_gauss[i - 16000]

                _data.loc[i, 'down_pkt_count'] += nn0_gauss[i - 16000]
                _data.loc[i, 'down_payload_count'] += nn0_gauss[i - 16000]
                _data.loc[i, 'down_payload_min'] += nn0_gauss[i - 16000]
                _data.loc[i, 'down_payload_max'] += nn0_gauss[i - 16000]
        for i in range(24000, 32000):
            _data.loc[i, 'duration'] += nn0[i - 24000]
            if _data.iloc[i, 0] != 0:
                _data.loc[i, 'up_time_interval_mean'] += nn0[i - 24000]
                _data.loc[i, 'up_time_interval_min'] += nn0[i - 24000]
                _data.loc[i, 'up_time_interval_max'] += nn0[i - 24000]
                _data.loc[i, 'up_time_interval_std'] += nn0[i - 24000]

                _data.loc[i, 'up_payload_mean'] += nn0_gauss[i - 24000]
                _data.loc[i, 'up_payload_std'] += nn0_gauss[i - 24000]

                _data.loc[i, 'up_pkt_count'] += nn0_gauss[i - 24000]
                _data.loc[i, 'up_payload_count'] += nn0_gauss[i - 24000]
                _data.loc[i, 'up_payload_min'] += nn0_gauss[i - 24000]
                _data.loc[i, 'up_payload_max'] += nn0_gauss[i - 24000]
            else:
                _data.loc[i, 'down_time_interval_mean'] += nn0[i - 24000]
                _data.loc[i, 'down_time_interval_min'] += nn0[i - 24000]
                _data.loc[i, 'down_time_interval_max'] += nn0[i - 24000]
                _data.loc[i, 'down_time_interval_std'] += nn0[i - 24000]

                _data.loc[i, 'down_payload_mean'] += nn0_gauss[i - 24000]
                _data.loc[i, 'down_payload_std'] += nn0_gauss[i - 24000]

                _data.loc[i, 'down_pkt_count'] += nn0_gauss[i - 24000]
                _data.loc[i, 'down_payload_count'] += nn0_gauss[i - 24000]
                _data.loc[i, 'down_payload_min'] += nn0_gauss[i - 24000]
                _data.loc[i, 'down_payload_max'] += nn0_gauss[i - 24000]
        _data.to_csv('dataset/multi/data_ICS_6_TT_multi_noise_{}.csv'.format(kk / 20), index=False)
    print('Add_multi_Noise0_Down')


get_noisy_multi_0(data0)

for i in range(1, 21):
    data = pd.read_csv('dataset/multi/data_ICS_6_TT_multi_noise_{}.csv'.format(i / 20))
    for j in range(8000, 32000):
        data.iloc[j, 27:28] = 1
    data.to_csv('dataset/binary/data_ICS_6_TT_binary_noise_{}.csv'.format(i / 20), index=False)
print('Add_binary_Noise0_Down')
