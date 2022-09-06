from PIL import Image
import cv2
import os
import glob
import numpy as np
from torchvision import transforms
from manydepth.test_simple import test_simple, _parse_args

import manydepth.ransac as ransac
from manydepth.ransac import RANSAC, LinearRegressor


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


def scale_depth(pred_depth, gt_depth, scaling):
    if scaling == "no_scaling":
        ratio = 1.0
    elif scaling == "median_scaling":
        ratio = np.median(gt_depth) / np.median(pred_depth)
    elif scaling == "ransac_leastsquares":
        regressor = RANSAC(
            n=100000,
            d=1000, k=10, t=100.0, model=LinearRegressor(), loss=ransac.square_error_loss, metric=ransac.mean_square_error,
        )
        regressor.fit(pred_depth.reshape(-1, 1), gt_depth.reshape(-1, 1))
        return regressor.predict(pred_depth.reshape(-1, 1)).reshape(gt_depth.shape)
    else:
        raise ValueError("Invalid scaling algorithm")
    return pred_depth * ratio


if __name__ == '__main__':
    args = parse_args_multiple()
    if args.dir_gt_depth == "":
        print("No ground truth depths are provided.")
    dir_gt_depth_vis = args.dir_gt_depth + "_vis"
    os.makedirs(dir_gt_depth_vis, exist_ok=True)
    image_paths = sorted(glob.glob(args.dir_img + "/*"))
    dir_pred_depth = args.dir_img + "_pred_depth"
    os.makedirs(dir_pred_depth, exist_ok=True)
    # CAUTION: gt depth paths should be stored in the proper directory and matched with the image_paths

    scale_vis = 3
    for source, target in zip(image_paths[:-1], image_paths[1:]):
        args = set_source_target(args, source, target)
        pred_depth = test_simple(args)
        if args.dir_gt_depth != "":
            # scaling
            gt_depth_path = args.dir_gt_depth + "/" + os.path.basename(target)
            _gt_depth = load_image(gt_depth_path)
            cv2.imwrite(dir_gt_depth_vis + "/" + os.path.basename(target), scale_vis * _gt_depth.numpy().squeeze().astype(np.uint16))
            # gt_depth = (_gt_depth / 1000).numpy().squeeze()  # mm -> m

            pred_depth = cv2.resize(pred_depth.numpy().squeeze(), dsize=(_gt_depth.shape[-1], _gt_depth.shape[-2]))
            _pred_depth = pred_depth * 1000  # m -> mm
            _pred_depth = scale_depth(_pred_depth, _gt_depth.numpy().squeeze(), "ransac_leastsquares")
            cv2.imwrite(dir_pred_depth + "/" + os.path.basename(target), scale_vis * _pred_depth.astype(np.uint16))
