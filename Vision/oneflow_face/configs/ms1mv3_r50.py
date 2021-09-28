from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.loss = "cosface"
config.network = "r50"
config.resume =True
config.output ="eager_third"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.1  # batch size is 512





config.ofrecord_path="/data/insightface/ms1m-retinaface-t1/ofrecord"
config.num_classes = 93431
config.num_image = 5179510
config.num_epoch = 25-6
config.warmup_epoch = -1
config.decay_epoch = [10-6, 16-6, 22-6]
#config.decay_epoch = [7, 13]

config.ofrecord_part_num=32
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
config.val_targets = []
config.val_image_num={"lfw":12000,"cfp_fp":14000,"agedb_30":12000}


