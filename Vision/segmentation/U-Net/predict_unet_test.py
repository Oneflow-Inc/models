import glob
import numpy as np
import oneflow
import os
import cv2
from unet import UNet
import argparse


def _parse_args():
    parser = argparse.ArgumentParser("Flags for test U-Net")
    parser.add_argument(
        "--checkpoint", type=str, default='./checkpoints', help="checkpoint"
    )
    parser.add_argument("--Test_Data_path", type=str,
                        default='test_image/', help="Test_Data_path")
    parser.add_argument("--save_res_path", type=str,
                        default="./predict_image/test.png", help="save_res_path")
    return parser.parse_args()


def main(args):
    device = oneflow.device('cuda' if oneflow.cuda.is_available() else 'cpu')
    net = UNet(n_channels=1, n_classes=1, bilinear=False)
    net.to(device=device)
    checkpoint = oneflow.load(args.checkpoint)
    net.load_state_dict(checkpoint['net'])
    net.eval()
    Test_Data_path = args.Test_Data_path
    tests_path = glob.glob(Test_Data_path + '*.png')

if __name__ == "__main__":
    device = oneflow.device("cuda" if oneflow.cuda.is_available() else "cpu")
    net = UNet(n_channels=1, n_classes=1, bilinear=False)
    net.to(device=device)
    checkpoint = oneflow.load("./checkpoints")
    net.load_state_dict(checkpoint["net"])
    net.eval()
    Test_Data_path = "test_image/"
    tests_path = glob.glob(Test_Data_path + "*.png")
    print("begin look")
    for test_path in tests_path:
        save_res_path = args.save_res_path
        img = cv2.imread(test_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        print("test")
        img_tensor = oneflow.tensor(np.array(img))
        print(img_tensor)
        img_tensor = img_tensor.to(device=device, dtype=oneflow.float32)
        pred = net(img_tensor)
        pred = np.array(pred.data.cpu()[0])[0]
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        print(np.unique(pred))
        pred = pred.astype(np.float32)
        cv2.imwrite(save_res_path, pred)


if __name__ == '__main__':
    args = _parse_args()
    main(args)
