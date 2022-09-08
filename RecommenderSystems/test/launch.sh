image_tag=oneflowinc/oneflow:nightly-cuda11.2
raw_data_dir=/data
parquet_data_dir=/workspace/kaggle_parquet
script_path=/workspace/deepfm/deepfm_train_eval.py

cmd_install_oneflow="/workspace/test/install_oneflow.sh | tee install.log"
cmd_make_dataset="python3 /workspace/deepfm/tools/make_parquet.py --input_dir=$raw_data_dir --output_dir=${parquet_data_dir} | tee make_dataset.log"
cmd_test_deepfm="python3 /workspace/test/test_deepfm.py --data_dir=$parquet_data_dir --script_path=$script_path"

docker run --privileged --network=host --ipc=host --gpus=all \
    -d \
    -v $PWD:/workspace \
    -v /data/criteo_kaggle/dac:$raw_data_dir \
    -w /workspace \
    $image_tag \
    /bin/sh \
    -c "$cmd_install_oneflow ; $cmd_make_dataset ; $cmd_test_deepfm"
     
docker rm $(docker stop $(docker ps -a -q --filter ancestor=$image_tag --format="{{.ID}}"))
