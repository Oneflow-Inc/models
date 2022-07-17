import os
import numpy as np
import oneflow as flow
from oneflow.one_embedding import make_persistent_table_writer



tables = [os.path.join('persistent/persistent1', f'{i}-1') for i in range(1)]
with make_persistent_table_writer(tables, "2022-07-14-21-53-04-270525", flow.int64, flow.float, 64) as writer:
    keys = np.arange(63001).astype(np.int64)
    values = np.load('./hist_item_emb_attr.weight.npy').T
    writer.write(keys, values)

tables = [os.path.join('persistent/persistent2', f'{i}-1') for i in range(1)]
with make_persistent_table_writer(tables, "2022-07-14-21-53-04-270525", flow.int64, flow.float, 64) as writer:
    keys = np.arange(801).astype(np.int64)
    values = np.load('./hist_cat_emb_attr.weight.npy').T
    writer.write(keys, values)

tables = [os.path.join('persistent/persistent3', f'{i}-1') for i in range(1)]
with make_persistent_table_writer(tables, "2022-07-14-21-53-04-270525", flow.int64, flow.float, 64) as writer:
    keys = np.arange(63001).astype(np.int64)
    values = np.load('./target_item_seq_emb_attr.weight.npy').T
    writer.write(keys, values)

tables = [os.path.join('persistent/persistent4', f'{i}-1') for i in range(1)]
with make_persistent_table_writer(tables, "2022-07-14-21-53-04-270525", flow.int64, flow.float, 64) as writer:
    keys = np.arange(801).astype(np.int64)
    values = np.load('./target_cat_seq_emb_attr.weight.npy').T
    writer.write(keys, values)

tables = [os.path.join('persistent/persistent5', f'{i}-1') for i in range(1)]
with make_persistent_table_writer(tables, "2022-07-14-21-53-04-270525", flow.int64, flow.float, 64) as writer:
    keys = np.arange(63001).astype(np.int64)
    values = np.load('./target_item_emb_attr.weight.npy').T
    writer.write(keys, values)

tables = [os.path.join('persistent/persistent6', f'{i}-1') for i in range(1)]
with make_persistent_table_writer(tables, "2022-07-14-21-53-04-270525", flow.int64, flow.float, 64) as writer:
    keys = np.arange(801).astype(np.int64)
    values = np.load('./target_cat_emb_attr.weight.npy').T
    writer.write(keys, values)

tables = [os.path.join('persistent/persistent7', f'{i}-1') for i in range(1)]
with make_persistent_table_writer(tables, "2022-07-14-21-53-04-270525", flow.int64, flow.float, 1) as writer:
    keys = np.arange(63001).astype(np.int64)
    values = np.load('./item_b_attr.weight.npy').T
    writer.write(keys, values)