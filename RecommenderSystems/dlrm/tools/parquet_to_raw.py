from petastorm.reader import make_batch_reader
import glob
import numpy as np
import time

fields = ['label']
fields += ["I{}".format(i + 1) for i in range(13)]
fields += ["C{}".format(i + 1) for i in range(26)]

def parquet_to_raw(files, output_dir):
    with make_batch_reader(files, workers_count=1, shuffle_row_groups=False) as reader:
        lf = open(f'{output_dir}/label.bin', 'wb')
        df = open(f'{output_dir}/dense.bin', 'wb')
        sf = open(f'{output_dir}/sparse.bin', 'wb')
        dnf = open(f'{output_dir}/dense_norm.bin', 'wb')
        for rg in reader:
            rgdict = rg._asdict()
            rglist = [rgdict[field] for field in fields]
            label = rglist[0]
            lf.write(label.tobytes())
            dense = np.stack(rglist[1:14], axis=-1)
            df.write(dense.tobytes())
            dense_norm = np.log(dense + 1.0)
            dnf.write(dense_norm.tobytes())
            sparse = np.stack(rglist[14:40], axis=-1)
            sf.write(sparse.tobytes())
            print(label.shape, dense.shape, sparse.shape)
        lf.close(), df.close(), sf.close(), dnf.close()

trains = ['file://' + name for name in glob.glob('/RAID0/dlrm_parquet_int32/train/*.parquet')]
tests = ['file://' + name for name in glob.glob('/RAID0/dlrm_parquet_int32/test/*.parquet')]
vals = ['file://' + name for name in glob.glob('/RAID0/dlrm_parquet_int32/val/*.parquet')]

trains.sort()
tests.sort()
vals.sort()

parquet_to_raw(tests, "/RAID0/raw/test")
parquet_to_raw(vals, "/RAID0/raw/val")
parquet_to_raw(trains, "/RAID0/raw/train")
