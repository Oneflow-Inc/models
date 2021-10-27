#!/bin/bash

stage=-1  # Modify this to control to start from which stage

train_dir="data/wjs0_2mix/tr/"
dev_dir="data/wjs0_2mix/cv/"
test_dir="data/wjs0_2mix/tt/"
model_path='final.pth.tar'
dump_dir="sps_tas"

# exp tag
tag="" # tag for managing experiments.

ngpu=1


if [ -z ${tag} ]; then
  expdir=exp/convtasnet
else
  expdir=exp/train_${tag}
fi
mkdir -p ${expdir}

if [ $stage -le 1 ]; then
  echo "Stage 1: Generating .scp file including wav path for training&validation&testing data"
  ./nnet/create_scp.py \
  --dataPath ${train_dir} \
  --data "mix" \
  --scp_name "mix.scp"
  ./nnet/create_scp.py \
  --dataPath ${train_dir} \
  --data "s1" \
  --scp_name "spk1.scp"
  ./nnet/create_scp.py \
  --dataPath ${train_dir} \
  --data "s2" \
  --scp_name "spk2.scp"
  ./nnet/create_scp.py \
  --dataPath ${dev_dir} \
  --data "mix" \
  --scp_name "mix.scp"
  ./nnet/create_scp.py \
  --dataPath ${dev_dir} \
  --data "s1" \
  --scp_name "spk1.scp"
  ./nnet/create_scp.py \
  --dataPath ${dev_dir} \
  --data "s2" \
  --scp_name "spk2.scp"
  ./nnet/create_scp.py \
  --dataPath ${test_dir} \
  --data "mix" \
  --scp_name "mix.scp"
  ./nnet/create_scp.py \
  --dataPath ${test_dir} \
  --data "s1" \
  --scp_name "spk1.scp"
  ./nnet/create_scp.py \
  --dataPath ${test_dir} \
  --data "s2" \
  --scp_name "spk2.scp"
fi

if [ $stage -le 2 ]; then
  echo "Stage 2: Training"
  ./nnet/train.py \
  --model_path ${model_path} \
  --save_folder ${expdir} \
  > ${expdir}/train.log 2>&1 
fi

if [ $stage -le 3 ]; then
  echo "Stage 3: Separate speech using the trained Conv-TasNet"
  ./nnet/separate.py \
  --model_path ${expdir}/$model_path \
  --dump-dir ${dump_dir} \
  --input ${test_dir}/mix.scp > ${expdir}/separate.log 2>&1 & 
fi

if [ $stage -le 4 ]; then
  echo "Stage 4: Generating .scp file including wav path for the seperated audio"
  ./nnet/create_scp.py \
  --dataPath ${dump_dir} \
  --data "spk1" \
  --scp_name "s1.scp"
  ./nnet/create_scp.py \
  --dataPath ${dump_dir} \
  --data "spk2" \
  --scp_name "s2.scp"
fi

if [ $stage -le 5 ]; then
  echo "Stage 5: Evaluate separation performance"
  ./nnet/compute_si_snr.py \
  --ref_scp ${test_dir}/spk1.scp,${test_dir}/spk2.scp \
  --sep_scp ${dump_dir}/s1.scp,${dump_dir}/s2.scp > ${expdir}/evaluate.log 2>&1 & 
fi