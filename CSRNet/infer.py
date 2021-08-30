import argparse
import oneflow as flow
import transforms.spatial_transforms as ST
from matplotlib import pyplot as plt
from image import *
from model import CSRNet
from matplotlib import cm as c

parser = argparse.ArgumentParser(description="Oneflow CSRNet")
parser.add_argument(
    "modelPath", metavar="MODELPATH", type=str, help="path to bestmodel"
)
parser.add_argument("picPath", metavar="PICPATH", type=str, help="path to testPic")
parser.add_argument(
    "picDensity", metavar="PICDensity", type=str, help="path to PICDensity"
)


def main():
    transform = ST.Compose(
        [
            ST.ToNumpyForVal(),
            ST.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    global args
    args = parser.parse_args()
    model = CSRNet()
    model = model.to("cuda")
    # checkpoint = flow.load('checkpoint/Shanghai_BestModelA/shanghaiA_bestmodel')
    checkpoint = flow.load(args.modelPath)
    model.load_state_dict(checkpoint)
    img = transform(Image.open(args.picPath).convert("RGB"))
    img = flow.Tensor(img)
    img = img.to("cuda")
    output = model(img.unsqueeze(0))
    print("Predicted Count : ", int(output.detach().to("cpu").sum().numpy()))
    temp = output.detach().reshape((output.detach().shape[2], output.detach().shape[3]))
    temp = temp.numpy()
    plt.title("Predicted Count")
    plt.imshow(temp, cmap=c.jet)
    plt.show()
    temp = h5py.File(args.picDensity, "r")
    temp_1 = np.asarray(temp["density"])
    plt.title("Original Count")
    plt.imshow(temp_1, cmap=c.jet)
    print("Original Count : ", int(np.sum(temp_1)) + 1)
    plt.show()
    print("Original Image")
    plt.title("Original Image")
    plt.imshow(plt.imread(args.picPath))
    plt.show()


if __name__ == "__main__":
    main()
