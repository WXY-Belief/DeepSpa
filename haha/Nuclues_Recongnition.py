import os
import time
import cv2
import mmcv
import copy
import numpy as np
from skimage import io
from .deep_learing_model.mmseg.apis import inference_segmentor, init_segmentor
from .deep_learing_model.mmseg.datasets import result_to_inst
from skimage.segmentation import find_boundaries
from scipy import ndimage
import pandas as pd
from PIL import Image


def find_dapi_img(path):
    for file_name in os.listdir(path):
        if os.path.join(path, file_name).lower().endswith(('.png', '.jpg', '.tif')):
            return os.path.join(path, file_name)


# Draw the outline of nucleus
def make_outline_overlay(RGB_data, predictions):
    boundaries = np.zeros_like(predictions)
    overlay_data = copy.copy(RGB_data)

    for img in range(predictions.shape[0]):
        boundary = find_boundaries(predictions[img, :, :], connectivity=1, mode='inner')
        boundaries[img, boundary > 0] = 1

    overlay_data[boundaries > 0, :] = (0, 255, 0)

    return overlay_data


def get_cell_center_and_area(pre_result, save_path):
    areas = []  # save the area of each nucleus
    centers = np.zeros((pre_result.max(), 2), 'int')  # save the center coordinate of each nucleus

    slices = ndimage.find_objects(pre_result)
    for i, si in enumerate(slices):
        if si is not None:
            sr, sc = si
            yi, xi = np.nonzero(pre_result[sr, sc] == (i + 1))
            yi = yi.astype(np.int32)
            xi = xi.astype(np.int32)
            ymed = np.median(yi)
            xmed = np.median(xi)
            imin = np.argmin((xi - xmed) ** 2 + (yi - ymed) ** 2)
            xmed = xi[imin]
            ymed = yi[imin]
            centers[i, 0] = ymed + sr.start
            centers[i, 1] = xmed + sc.start
            areas.append(len(yi))
    info = pd.DataFrame()
    info["row"] = centers[:, 0]
    info["col"] = centers[:, 1]
    info["area"] = areas
    info["cell_index"] = [int(i) for i in range(1, info.shape[0] + 1)]
    info.to_csv(os.path.join(save_path, "cell_center.csv"), sep=",", header=True, index=False)


# Detected nucleus
def predict(img_path, save_path, device, threshold):
    config = "./haha/deep_learing_model/configs/config.py"  # config file
    checkpoint = "./haha/deep_learing_model/final_model/dsb_tissue_56000.pth"
    # convert dapi images to format of RGB
    dapi_rgb = Image.open(img_path).convert("RGB")
    dapi_rgb_path = os.path.join(save_path, "DAPI_RGB.PNG")
    dapi_rgb.save(dapi_rgb_path)

    prefix = dapi_rgb_path.split('/')[-1].split(".")[0]
    cfg = mmcv.Config.fromfile(config)
    model = init_segmentor(
        cfg,
        checkpoint,
        device=device,
        revise_checkpoint=[(r'^module\.', ''), ('model.', '')])
    result = inference_segmentor(model, dapi_rgb_path)
    pred = result_to_inst(result[0], threshold=threshold)[0]

    np.save(os.path.join(save_path, "pre_result.npy"), pred)

    ori_img = io.imread(dapi_rgb_path)
    overlay = make_outline_overlay(np.expand_dims(ori_img, 0), np.expand_dims(pred, 0))  # make contour of nucleus
    cv2.imwrite(os.path.join(save_path, prefix + "_boundary.PNG"), overlay[0])

    get_cell_center_and_area(pred, save_path)


# filter nuleus
def filter_nucleus(save_path, mode, top_value, bottom_value):
    cell_info = pd.read_csv(os.path.join(save_path, "cell_center.csv"), sep=",", header=0)
    if mode == 1:
        cell_info = cell_info[
            (cell_info["area"] > (bottom_value ** 2) * 3.14) & (cell_info["area"] < (top_value ** 2) * 3.14)]
    else:
        cell_info.sort_values(by="area", inplace=True, ascending=True)
        cell_num = cell_info.shape[0]
        cell_info = cell_info.iloc[int(cell_num / 100 * bottom_value):cell_num - int(cell_num / 100 * top_value)]

    cell_info.to_csv(os.path.join(save_path, "filtered_cell_center.csv"), sep=",", header=True, index=False)


# detecting and filtering nucleus
def nucleus_recongnition(data_path, output_path, device, mode, top_value, bottom_value, threshold):
    device = "cuda" if device == "GPU" else "cpu"
    star_time = time.time()
    all_section = os.listdir(data_path)
    for item in all_section:
        dapi_path = find_dapi_img(os.path.join(data_path, item))
        save_path = os.path.join(output_path, item, "1_nucleus_recongnition_result")
        os.makedirs(save_path, exist_ok=True)

        # Detected nucleus and determine the center of cell
        predict(dapi_path, save_path, device, threshold)

        # filter nuleus
        filter_nucleus(save_path, mode, top_value, bottom_value)

    print(f" detecting nucleus is finished, runtime：{time.time() - star_time}")
