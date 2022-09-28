import json
import os

import cv2
import argparse
import oneflow as of
import flowvision.transforms as ft


def _parse_args():

    parser = argparse.ArgumentParser("flags for predict model")

    parser.add_argument(
        "--model_layer",
        type=int,
        default=161,
        help="model layer",
    )

    parser.add_argument(
        "--pth_path",
        type=str,
        default=" ",
        help="model pth path",
    )

    parser.add_argument(
        "--image_test_json",
        type=str,
        default='submit_example.json',
        help="input image test json file")

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="model device",
    )

    return parser.parse_args()

def main(args):

    model_layer = args.model_layer

    if model_layer == 121:
        from DenseNet import DenseNet121_pre
        model = DenseNet121_pre(pretrained=False)
    elif model_layer == 169:
        from DenseNet import DenseNet169_pre
        model = DenseNet169_pre(pretrained=False)
    elif model_layer == 201:
        from DenseNet import DenseNet201_pre
        model = DenseNet201_pre(pretrained=False)
    else:
        from DenseNet import DenseNet161_pre
        model = DenseNet161_pre(pretrained=False)

    pth_path = args.pth_path
    if not os.path.exists(pth_path):
        print('pth path is not exist')
        return -1

    model.load_state_dict(of.load(pth_path))

    device = args.device
    model.to(device)
    model.eval()

    image_test_json = args.image_test_json
    if not os.path.exists(image_test_json):
        print('image test json is not exist')
        return -1

    file = open(image_test_json)
    infos = json.load(file)
    annotations = infos['annotations']

    size = 224
    transforms = ft.Compose([
                ft.ToTensor(),
                ft.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]
            )

    result = {}
    ann = []
    len1 = len(annotations)
    print('The total number:', len1)
    for temp in annotations:
        imgfile = temp['filename']
        img = cv2.imread(imgfile)
        img = cv2.blur(img, ksize=(9, 9))
        img = transforms(img)
        img = of.reshape(img, (1, 3, size, size))
        img = img.to('cuda')
        pred = model(img)
        _, indices = of.topk(pred, k=1, dim=1)
        re = indices.item()
        ann.append({'filename': imgfile, 'label': int(re)})

    result['annotations'] = ann
    with open("{}.json".format(pth_path), "w", encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))


if __name__ == '__main__':
    args = _parse_args()
    main(args)
