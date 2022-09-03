from PIL import Image
import cv2
import os
import glob
import numpy as np
from torchvision import transforms
from manydepth.test_simple import test_simple, _parse_args


def parse_args_multiple():
    parser = _parse_args()
    parser.add_argument("--dir_img", type=str, help="Data in dir should be sorted by their names")
    parser.add_argument("--dir_gt_depth", type=str, help="Data in dir should be sorted by their names and be matched with `dir_img`", default="")
    return parser.parse_args()


def set_source_target(args, source, target):
    args.source_image_path = source
    args.target_image_path = target
    return args


def load_image(image_path):
    image = Image.open(image_path)
    image = transforms.ToTensor()(image)
    return image


def scale_depth(pred_depth, gt_depth):
    ratio = gt_depth.median() / pred_depth.median()
    return pred_depth * ratio


if __name__ == '__main__':
    args = parse_args_multiple()
    if args.dir_gt_depth == "":
        print("No ground truth depths are provided.")
    image_paths = sorted(glob.glob(args.dir_img + "/*"))
    # CAUTION: gt depth paths should be stored in the proper directory and matched with the image_paths
    # CAUTION: gt depth paths should be stored in the proper directory and matched with the image_paths
    for source, target in zip(image_paths[:-1], image_paths[1:]):
        args = set_source_target(args, source, target)
        pred_depth = test_simple(args)
        if args.dir_gt_depth != "":
            dir_pred_depth = args.dir_img + "_pred_depth"
            os.makedirs(dir_pred_depth, exist_ok=True)
            # scaling
            gt_depth_path = args.dir_gt_depth + "/" + os.path.basename(target)
            _gt_depth = load_image(gt_depth_path)
            gt_depth = _gt_depth / 1000  # mm -> m
            pred_depth = scale_depth(pred_depth, gt_depth)
            _pred_depth = pred_depth * 1000  # m -> mm
            pred_depth_img = cv2.resize(_pred_depth.numpy().astype(np.uint16).squeeze(), dsize=(_gt_depth.shape[-1], _gt_depth.shape[-2]))
            cv2.imwrite(dir_pred_depth + "/" + os.path.basename(target), pred_depth_img)
