import glob
import numpy as np
import oneflow
import os
import cv2
from unet import UNet

if __name__ == "__main__":
    device = oneflow.device('cuda' if oneflow.cuda.is_available() else 'cpu')
    net = UNet(n_channels=1, n_classes=1, bilinear=False)
    net.to(device=device)
    checkpoint = oneflow.load('./checkpoints')
    net.load_state_dict(checkpoint['net'])
    net.eval()
    Test_Data_path = 'test_image/'
    tests_path = glob.glob(Test_Data_path + '*.png')
    print("begin look")
    for test_path in tests_path:
        save_res_path = "./predict_image/1.png"
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
