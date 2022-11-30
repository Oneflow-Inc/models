docker run -it --rm --runtime=nvidia --privileged \
  --network host --gpus=all \
  --ipc=host \
  -v /RAID0:/RAID0 \
  oneflow-dlrm:0.1 \
  bash
