import os

import pandas as pd


def gen_index_map(column, offset=0):
    index_map = {origin: index + offset
                 for index, origin in enumerate(column.drop_duplicates())}
    return index_map


class AirDataset:
    def __init__(self, data_df, sensor_index_col='sensor_index'):
        self.data = data_df
        self.min_v, self.max_v = self.data.min(), self.data.max()
        self.data = (self.data - self.min_v) / (self.max_v - self.min_v)
        self.data = self.data.reset_index()

        self.sensor_index_col = sensor_index_col

    def denormalize(self, x, feat):
        min_v, max_v = self.min_v[feat], self.max_v[feat]
        return x * (max_v - min_v) + min_v

    def construct_set(self, start_prop, end_prop, his_len, pre_len, feat_name):
        set = {'sensor_index': [], 'src': [], 'trg': []}
        for sensor_index, group in self.data.groupby(self.sensor_index_col):
            full_seq = group[feat_name]
            full_len = full_seq.shape[0]
            start_i, end_i = int(full_len * start_prop), int(full_len * end_prop)
            sequence = full_seq[start_i:end_i]

            for i in range(sequence.shape[0] - his_len - pre_len + 1):
                span = sequence.iloc[i:i+his_len+pre_len]
                # span = span.interpolate(limit=2, limit_direction='forward', limit_area='inside')
                if span.isna().any():
                    continue
                span = span.to_list()

                set['sensor_index'].append(sensor_index)
                set['src'].append(span[:his_len])
                set['trg'].append(span[-1])

            # TODO: remove this line
            if sensor_index >= 3: break
        return set


class KrakowDataset(AirDataset):
    def __init__(self):
        sensor_coor_df = pd.read_csv(os.path.join('dataset', 'Krakow-airquality', 'sensor_locations.csv'))
        id2index = gen_index_map(sensor_coor_df['id'])

        raw_data_list = ['april-2017', 'august-2017', 'december-2017', 'february-2017', 'january-2017',
                                        'july-2017', 'june-2017', 'march-2017', 'may-2017', 'november-2017', 'october-2017', 'september-2017']
        raw_data = pd.concat([pd.read_csv(os.path.join('dataset', 'Krakow-airquality', '{}.csv'.format(name)), parse_dates=[0])
                              for name in raw_data_list])
        df_list = []
        name_col = ['temperature', 'humidity', 'pressure', 'pm1', 'pm25', 'pm10']
        for sensor_id in sensor_coor_df['id']:
            this_sensor = raw_data[[col for col in raw_data.columns
                                    if col in ['UTC time'] + ['{}_{}'.format(sensor_id, n) for n in name_col]]]
            this_sensor = this_sensor.copy()
            this_sensor.columns = ['UTC time'] + name_col
            this_sensor['sensor_index'] = id2index[sensor_id]
            df_list.append(this_sensor)
        data_df = pd.concat(df_list).set_index(['sensor_index', 'UTC time']).sort_index()

        super().__init__(data_df)
