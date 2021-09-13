import argparse
import os
import numpy as np
import time
import pickle
import oneflow as flow
import oneflow.optim as optim
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

from utils.of_data_utils import NumpyDataLoader, ValDatasetFromFolder
from utils.of_loss import GeneratorLoss
from models.of_model import Generator, Discriminator

parser = argparse.ArgumentParser(description="Train Super Resolution Models")
parser.add_argument("--data_dir", default="./data/VOC", type=str, help="data root")
parser.add_argument("--hr_size", default=88, type=int, help="training images crop size")
parser.add_argument("--gpu_ids", type=str, default="0")
parser.add_argument(
    "--upscale_factor",
    default=4,
    type=int,
    choices=[2, 4, 8],
    help="super resolution upscale factor",
)
parser.add_argument("--num_epochs", default=1, type=int, help="train epoch number")
parser.add_argument(
    "--train_mode",
    default="GPU",
    type=str,
    choices=["GPU", "CPU"],
    help="using GPU or CPU",
)
parser.add_argument("--batch_size", type=int, default=256, required=False)
parser.add_argument("--load_checkpoint_G", type=str, default="", help="load checkpoint")
parser.add_argument("--load_checkpoint_D", type=str, default="", help="load checkpoint")

parser.add_argument(
    "--save_path", type=str, default="./srgan", help="save results root dir"
)
parser.add_argument(
    "--vgg_path",
    type=str,
    default="./vgg_imagenet_pretrain_model/vgg16_oneflow_model",
    help="vgg pretrained weight path",
)
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument(
    "--b1",
    type=float,
    default=0.5,
    help="adam: decay of first order momentum of gradient",
)
parser.add_argument(
    "--b2",
    type=float,
    default=0.999,
    help="adam: decay of first order momentum of gradient",
)


def to_numpy(x, mean=True):
    if mean:
        x = flow.mean(x)

    return x.numpy()


def to_tensor(x, grad=True, dtype=flow.float32):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    return flow.Tensor(x, requires_grad=grad, dtype=dtype)


def save_obj(obj, name):
    with open(name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    print("Save {} done.".format(name + ".pkl"))


if __name__ == "__main__":

    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids

    HR_SIZE = opt.hr_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    TRAIN_MODE = True if opt.train_mode == "GPU" else False
    lr_size = opt.hr_size // opt.upscale_factor
    modes = ["val", "train", "test"]
    train_data = NumpyDataLoader(
        dataset_root=opt.data_dir,
        mode=modes[1],
        hr_size=HR_SIZE,
        lr_size=lr_size,
        batch_size=opt.batch_size,
    )
    val_data = ValDatasetFromFolder(opt.data_dir, mode=modes[0], upscale_factor=4)
    start_t = time.time()

    netG = Generator(UPSCALE_FACTOR)
    print("# generator parameters:", sum(param.numel() for param in netG.parameters()))
    netD = Discriminator()
    print(
        "# discriminator parameters:", sum(param.numel() for param in netD.parameters())
    )

    generator_criterion = GeneratorLoss(opt.vgg_path)
    bce = flow.nn.BCEWithLogitsLoss()
    end_t = time.time()
    print("init time : {}".format(end_t - start_t))

    if TRAIN_MODE:
        netG.to("cuda")
        netD.to("cuda")
        generator_criterion.to("cuda")
        bce.to("cuda")

    if opt.load_checkpoint_G != "":
        netG.load_state_dict(flow.load(opt.load_checkpoint_G))
        print("netG")
    if opt.load_checkpoint_D != "":
        netD.load_state_dict(flow.load(opt.load_checkpoint_D))

    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    results = {
        "d_loss": [],
        "g_loss": [],
        "d_score": [],
        "g_score": [],
        "psnr": [],
        "ssim": [],
    }

    for epoch in range(0, NUM_EPOCHS):
        train_bar = tqdm(train_data)
        running_results = {
            "batch_sizes": 0,
            "d_loss": 0,
            "g_loss": 0,
            "d_score": 0,
            "g_score": 0,
        }
        netG.train()
        netD.train()
        for idx in range(len(train_data)):
            target, data = train_data[idx]
            batch_size = data.shape[0]
            running_results["batch_sizes"] += batch_size
            z = to_tensor(data, False)
            real_img = to_tensor(target, False)
            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################

            real_img = real_img.to("cuda")
            z = z.to("cuda")
            fake_img = netG(z)

            real_out = netD(real_img)
            fake_out = netD(fake_img.detach())
            label1 = to_tensor(
                np.random.rand(batch_size, 1) * 0.25 + 0.85, False, dtype=flow.float32
            ).to("cuda")
            label0 = to_tensor(
                np.random.rand(batch_size, 1) * 0.15, False, dtype=flow.float32
            ).to("cuda")
            d_loss = bce(fake_out, label0) + bce(real_out, label1)

            d_loss.backward()
            optimizerD.step()
            optimizerD.zero_grad()

            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################

            fake_img_0 = netG(z)
            fake_out_0 = netD(fake_img_0)
            g_loss = generator_criterion(fake_out_0, fake_img_0, real_img)
            g_loss.backward()
            optimizerG.step()
            optimizerG.zero_grad()

            fake_out = flow.mean(fake_out)
            real_out = flow.mean(fake_out)
            # loss for current batch before optimization
            running_results["g_loss"] += g_loss.numpy() * batch_size
            running_results["d_loss"] += d_loss.numpy() * batch_size
            running_results["d_score"] += real_out.numpy() * batch_size
            running_results["g_score"] += fake_out.numpy() * batch_size

            train_bar.set_description(
                desc="[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f"
                % (
                    epoch,
                    NUM_EPOCHS,
                    running_results["d_loss"] / running_results["batch_sizes"],
                    running_results["g_loss"] / running_results["batch_sizes"],
                    running_results["d_score"] / running_results["batch_sizes"],
                    running_results["g_score"] / running_results["batch_sizes"],
                )
            )

        netG.eval()
        out_path = os.path.join(opt.save_path, "SRF_" + str(UPSCALE_FACTOR))
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        with flow.no_grad():
            val_bar = tqdm(val_data)
            valing_results = {
                "psnrs": 0,
                "ssims": 0,
                "psnr": 0,
                "ssim": 0,
                "batch_sizes": 0,
            }
            val_images = []
            print("val data", len(val_data))
            for idx in range(len(val_data)):
                val_hr, val_lr = val_data[idx]
                batch_size = val_lr.shape[0]
                valing_results["batch_sizes"] += batch_size
                lr = to_tensor(val_lr)
                hr = to_tensor(val_hr)
                lr = lr.to("cuda")
                hr = hr.to("cuda")
                sr = netG(lr)

                fake = sr[0].data * 255.0
                fake = fake.squeeze(0).permute(1, 2, 0)
                fake = fake.numpy().astype(np.uint8)

                _img = hr[0].data * 255
                _img = _img.squeeze(0).permute(1, 2, 0)
                _img = _img.numpy().astype(np.uint8)

                batch_psnr = peak_signal_noise_ratio(_img, fake)
                valing_results["psnrs"] += batch_psnr * batch_size
                valing_results["psnr"] = (
                    valing_results["psnrs"] / valing_results["batch_sizes"]
                )

                batch_ssim = structural_similarity(_img, fake, multichannel=True)
                valing_results["ssims"] += batch_ssim * batch_size
                valing_results["ssim"] = (
                    valing_results["ssims"] / valing_results["batch_sizes"]
                )
                val_bar.set_description(
                    desc="[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f"
                    % (valing_results["psnr"], valing_results["ssim"])
                )

        # save loss\scores\psnr\ssim
        results["d_loss"].append(
            running_results["d_loss"] / running_results["batch_sizes"]
        )
        results["g_loss"].append(
            running_results["g_loss"] / running_results["batch_sizes"]
        )
        results["d_score"].append(
            running_results["d_score"] / running_results["batch_sizes"]
        )
        results["g_score"].append(
            running_results["g_score"] / running_results["batch_sizes"]
        )
        results["psnr"].append(valing_results["psnr"])
        results["ssim"].append(valing_results["ssim"])

    # save model parameters
    flow.save(
        netG.state_dict(),
        os.path.join(opt.save_path, "netG_epoch_%d_%d" % (UPSCALE_FACTOR, epoch)),
    )
    flow.save(
        netD.state_dict(),
        os.path.join(opt.save_path, "netD_epoch_%d_%d" % (UPSCALE_FACTOR, epoch)),
    )
    save_obj(results, os.path.join(opt.save_path, "results"))
