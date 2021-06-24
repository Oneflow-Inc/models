import os
from PIL import Image, ImageOps
import numpy as np


def load_facades(mode="train"):
    data_path = "./data/facades"
    seed = np.random.randint(1024)
    if not os.path.exists(data_path):
        print("not Found Facades - start download")
        if not os.path.exists("data"):
            os.mkdir("data")

        # code to download the dataset
        import subprocess
        cmd = "bash ./utils/download_pix2pix_dataset.sh"
        subprocess.call(cmd)

    else:
        print("Found Facades - skip download")
        pass

    input_imgs, real_imgs = [], []
    if mode == "train":
        # train: 400, test: 106, val:100
        modes = ["train", "val"]
    else:
        modes = ["test"]

    for mode in modes:
        for d in os.listdir(os.path.join(data_path, mode)):
            d = os.path.join(data_path, mode, d)
            img = np.asarray(Image.open(d))
            real_img = Image.fromarray(img[:, :256, :])
            input_img = Image.fromarray(img[:, 256:, :])

            # resize to 286 x 286 x 3, and randomly crop to 256 x 256 x 3
            r1, r2 = np.random.randint(30, size=2)
            real_img = real_img.resize((256 + 30, 256 + 30))
            input_img = input_img.resize((256 + 30, 256 + 30))
            real_img = real_img.crop((r1, r2, r1 + 256, r2 + 256))
            input_img = input_img.crop((r1, r2, r1 + 256, r2 + 256))

            if np.random.rand() > 0.5:
                # random mirroring
                real_img = ImageOps.mirror(real_img)
                input_img = ImageOps.mirror(input_img)

            real_imgs.append(np.asarray(real_img))
            input_imgs.append(np.asarray(input_img))

    input_imgs = np.array(input_imgs).transpose(0, 3, 1, 2).astype(np.float32)
    real_imgs = np.array(real_imgs).transpose(0, 3, 1, 2).astype(np.float32)
    # normalizing the images to [-1, 1]
    input_imgs = input_imgs / 127.5 - 1
    real_imgs = real_imgs / 127.5 - 1

    np.random.seed(seed)
    np.random.shuffle(input_imgs)
    np.random.seed(seed)
    np.random.shuffle(real_imgs)
    return input_imgs, real_imgs
