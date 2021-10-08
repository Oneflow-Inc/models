import os
import glob
import argparse
import oneflow as flow
import transforms.spatial_transforms as ST
from image import *
from model import CSRNet

parser = argparse.ArgumentParser(description="Oneflow CSRNet")
parser.add_argument(
    "modelPath", metavar="MODELPATH", type=str, help="path to bestmodel"
)
parser.add_argument("picSrc", metavar="picSRC", type=str, help="path to valPic")


def main():
    transform = ST.Compose(
        [
            ST.ToNumpyForVal(),
            ST.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    global args
    args = parser.parse_args()
    root = "./dataset/"
    # now generate the ShanghaiA's ground truth
    part_A_train = os.path.join(root, "part_A_final/train_data", "images")
    part_A_test = os.path.join(root, "part_A_final/test_data", "images")
    part_B_train = os.path.join(root, "part_B_final/train_data", "images")
    part_B_test = os.path.join(root, "part_B_final/test_data", "images")
    path_sets = []
    if args.picSrc == "part_A_test":
        path_sets = [part_A_test]
    elif args.picSrc == "part_B_test":
        path_sets = [part_B_test]
    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, "*.jpg")):
            img_paths.append(img_path)
    model = CSRNet()
    model = model.to("cuda")
    checkpoint = flow.load(args.modelPath)
    model.load_state_dict(checkpoint)
    MAE = []
    for i in range(len(img_paths)):
        img = transform(Image.open(img_paths[i]).convert("RGB"))
        img = np.asarray(img).astype(np.float32)
        img = flow.Tensor(img, dtype=flow.float32, device="cuda")
        img = img.to("cuda")
        gt_file = h5py.File(
            img_paths[i].replace(".jpg", ".h5").replace("images", "ground_truth"), "r"
        )
        groundtruth = np.asarray(gt_file["density"])
        with flow.no_grad():
            output = model(img.unsqueeze(0))
        mae = abs(output.sum().numpy() - np.sum(groundtruth))
        MAE.append(mae)
    avg_MAE = sum(MAE) / len(MAE)
    print("test result: MAE:{:2f}".format(avg_MAE))


if __name__ == "__main__":
    main()
