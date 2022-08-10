import os
import glob
import time
import argparse
import numpy as np
from petastorm.reader import make_batch_reader


def parquet_to_raw(files, output_dir):
    print(output_dir)
    fields = ['label']
    fields += ["C{}".format(i + 1) for i in range(39)]
    with make_batch_reader(files, workers_count=1, shuffle_row_groups=False) as reader:
        lf = open(f'{output_dir}/label.bin', 'wb')
        sf = open(f'{output_dir}/sparse_C39.bin', 'wb')
        for rg in reader:
            rgdict = rg._asdict()
            rglist = [rgdict[field] for field in fields]
            label = rglist[0]
            lf.write(label.tobytes())
            sparse = np.stack(rglist[1:40], axis=-1)
            sf.write(sparse.tobytes())
            print(label.shape, sparse.shape)
        lf.close(), sf.close()


if __name__ == "__main__":
    def str_list(x):
        return list(map(str, x.split(",")))
    parser = argparse.ArgumentParser(description="convert parquet dataset to oneflow row")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sub_sets", type=str_list, default="test,val,train")
    args = parser.parse_args()

    for sub_set in args.sub_sets:
        files = ['file://' + name for name in glob.glob(f'{args.input_dir}/{sub_set}/*.parquet')]
        files.sort()
        output_dir = os.path.join(args.output_dir, sub_set)
        os.system(f"mkdir -p {output_dir}")
        parquet_to_raw(files, output_dir)

