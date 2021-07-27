import argparse
import time
import numpy as np

from PIL import Image
import oneflow as flow
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

from models.of_model import Generator



parser = argparse.ArgumentParser(description="Test Single Image")
parser.add_argument(
    "--upscale_factor", default=4, type=int, help="super resolution upscale factor"
)
parser.add_argument(
    "--test_mode",
    default="GPU",
    type=str,
    choices=["GPU", "CPU"],
    help="using GPU or CPU",
)
parser.add_argument(
    "--image_path",
    default="data/Set14/LR_bicubic/X4/monarchx4.png",
    type=str,
    help="test low resolution image path",
)
parser.add_argument(
    "--hr_path",
    default="data/Set14/HR/monarch.png",
    type=str,
    help="test low resolution image path",
)
parser.add_argument(
    "--save_image",
    default="data/Set14/SR/X4/monarchx4-oneflow.png",
    type=str,
    help="super resolution image path",
)
parser.add_argument(
    "--model_path",
    default="netG_epoch_4_99",
    type=str,
    help="generator model epoch name",
)
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == "GPU" else False
IMAGE_NAME = opt.image_path
SAVE_IMAGE = opt.save_image
MODEL_NAME = opt.model_path

model = Generator(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model.to("cuda")
model.load_state_dict(flow.load(MODEL_NAME))


image0 = Image.open(IMAGE_NAME)
# oneflow
image = image0.copy().convert("RGB")
tensor = np.ascontiguousarray(image).astype("float32")
tensor = tensor / 255
tensor = flow.Tensor(tensor)
tensor1 = tensor.unsqueeze(0).permute(0, 3, 1, 2)

if TEST_MODE:
    image = tensor1.to("cuda")

start = time.process_time()
with flow.no_grad():
    out = model(image)
elapsed = time.process_time() - start
print("cost" + str(elapsed) + "s")

## oneflow
out_a = out[0].data.to("cpu") * 255
out_b = out_a.squeeze(0).permute(1, 2, 0)
_img = out_b.numpy().astype(np.uint8)
if opt.hr_path != "":
    HR_NAME = opt.hr_path
    image_hr = np.array(Image.open(HR_NAME))
    psnr = peak_signal_noise_ratio(image_hr, _img)
    ssim = structural_similarity(image_hr, _img, multichannel=True)
    print("psnr:{},ssim:{}".format(psnr, ssim))

out_img = Image.fromarray(_img)
out_img.save(SAVE_IMAGE, quality=95)
