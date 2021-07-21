## Preparing Kinetics-400

For more details, please refer to the official [website](https://deepmind.com/research/open-source/open-source-datasets/kinetics/). We provide scripts with documentations. Before we start, please make sure that the directory is located at `$TSN/data_tools/kinetics400/`.

### Prepare annotations
First of all, run the following script to prepare annotations.
```shell
bash download_annotations.sh
```

### Prepare videos
Then, use the following script to prepare videos. The codes are adapted from the [official crawler](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics). Note that this might take a long time.
```shell
bash download_videos.sh
```

### Extract frames
Now it is time to extract frames from videos. 
If you only want to play with RGB frames (since extracting optical flow can be both time-comsuming and space-hogging), consider running the following script to extract **RGB-only** frames.
```shell
bash extract_rgb_frames.sh
```


### Generate filelist
Run the follow scripts to generate filelist in the format of videos and rawframes, respectively.
```shell
# execute the command below when rawframes are ready
bash generate_rawframes_filelist.sh
```

### Folder structure
In the context of the whole project (for kinetics400 only)

```
TSN
├── tsn
├── tools
├── data
│   ├── kinetics400
│   │   ├── kinetics400_train_list_videos.txt
│   │   ├── kinetics400_val_list_videos.txt
│   │   ├── annotations
│   │   ├── rawframes_train
│   │   ├── rawframes_val


