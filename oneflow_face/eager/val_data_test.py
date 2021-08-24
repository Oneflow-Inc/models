import oneflow as flow

import sys
sys.path.append("..")
from eval import verification
from utils.utils_callbacks import CallBackVerification
from backbones import get_model
import os

import logging
#创建日志对象，并初始化
logger = logging.getLogger(__name__)
#设置输出日志的级别
logger.setLevel(level = logging.INFO)

val_targets = ["lfw","cfp_fp","agedb_30"]

val_image_num={"lfw":12000,"cfp_fp":14000,"agedb_30":12000}

ofrecord_path="/dev/shm/ofrecord"

#ofrecord_path="/dev/shm/ms1m-retinaface-t1/"
val=CallBackVerification(1,0,val_targets,ofrecord_path,image_nums=val_image_num)
backbone = get_model("r100").to("cuda")
backbone_pth = os.path.join("./", "oneflow_face_PReLU")
backbone.load_state_dict(flow.load(backbone_pth))
val(100,backbone)