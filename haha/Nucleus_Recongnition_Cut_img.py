import os
import cv2
import mmcv
import copy
import time
import numpy as np
import pandas as pd
from scipy import ndimage
from .deep_learing_model.mmseg.apis import inference_segmentor, init_segmentor
from .deep_learing_model.mmseg.datasets import result_to_inst
from .deep_learing_model.tools.test import update_legacy_cfg
from skimage.segmentation import find_boundaries
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
cut_size = 512  # size of cut img
config = "./haha/deep_learing_model/configs/config_cut_img.py"
checkpoint = "./haha/deep_learing_model/final_model/dsb_iter_70000.pth"


def make_dir(path, dir_name) -> str:
    os.makedirs(os.path.join(path, dir_name), exist_ok=True)
    return os.path.join(path, dir_name)


def find_dapi_img(path):
    for file_name in os.listdir(path):
        if os.path.join(path, file_name).lower().endswith(('.png', '.jpg', '.tif')):
            return os.path.join(path, file_name)


def cut_img(dapi_path, save_path) -> None:
    global row_size, col_size
    ori_img = Image.open(dapi_path).convert("RGB")
    row_size, col_size = ori_img.height, ori_img.width

    ori_img_512 = np.zeros((row_size + cut_size, col_size + cut_size, 3))
    ori_img_512[0:row_size, 0:col_size, :] = ori_img

    num_row, num_col = int(ori_img_512.shape[0] / cut_size), int(ori_img_512.shape[1] / cut_size)
    for i in range(0, num_row):
        for j in range(0, num_col):
            final_cut_img = ori_img_512[i * cut_size: (i + 1) * cut_size, j * cut_size: (j + 1) * cut_size, :]
            cv2.imwrite(os.path.join(save_path, '{}_{}.PNG'.format(i + 1, j + 1)), final_cut_img)


# Draw the outline of nucleus
def make_outline_overlay(RGB_data, predictions):
    boundaries = np.zeros_like(predictions)
    overlay_data = copy.copy(RGB_data)

    for img in range(predictions.shape[0]):
        boundary = find_boundaries(predictions[img, :, :], connectivity=1, mode='inner')
        boundaries[img, boundary > 0] = 1

    overlay_data[boundaries > 0, :] = (0, 255, 0)

    return overlay_data


def predict(base_dir, img_path, prediction_result_path, boundary_result_path, device, threshold=0.5):
    infos = pd.DataFrame()
    boundery_img = np.zeros((row_size + cut_size, col_size + cut_size, 3))
    image_names, imgs = os.listdir(img_path), []
    cut_img_cell_center_path = make_dir(base_dir, "cut_img_cell_center")

    # build the model from a config file and a checkpoint file
    cfg = mmcv.Config.fromfile(config)
    cfg = update_legacy_cfg(cfg)
    model = init_segmentor(
        cfg,
        checkpoint,
        device=device,
        revise_checkpoint=[(r'^module\.', ''), ('model.', '')])

    for image_name in image_names:
        prefix = image_name.split('.')[0]
        image = os.path.join(img_path, image_name)

        result = inference_segmentor(model, image)
        pred = result_to_inst(result[0], threshold=threshold)[0]
        np.save(os.path.join(prediction_result_path, prefix + ".npy"), pred)

        ori_img = cv2.imread(image)
        overlay = make_outline_overlay(np.expand_dims(ori_img, 0), np.expand_dims(pred, 0))
        cv2.imwrite(os.path.join(boundary_result_path, image_name + ".PNG"), overlay[0])

        # determine cell center
        cut_img_cell_center = get_cell_center_and_area(pred, os.path.join(cut_img_cell_center_path, prefix + ".csv"))
        cut_img_cell_center["row"] += (int(float(prefix.split("_")[0])) - 1) * cut_size
        cut_img_cell_center["col"] += (int(float(prefix.split("_")[1])) - 1) * cut_size
        infos = pd.concat([infos, cut_img_cell_center])

        row_n, col_n = int(float(prefix.split("_")[0])) - 1, int(float(prefix.split("_")[1])) - 1
        boundery_img[cut_size * row_n: cut_size * (row_n + 1), cut_size * col_n: cut_size * (col_n + 1),
        :] = overlay[0]

    infos["cell_index"] = [int(i) for i in range(1, infos.shape[0] + 1)]
    infos.sort_values("col", inplace=True)
    infos.to_csv(os.path.join(base_dir, "cell_center.csv"), sep=",", header=True, index=False)

    boundery_img = boundery_img[:row_size - cut_size, :col_size - cut_size, :]
    cv2.imwrite(os.path.join(base_dir, "boundary.PNG"), boundery_img)


def get_cell_center_and_area(pred_result, save_path):
    areas = []  # save the area of each nucleus
    centers = np.zeros((pred_result.max(), 2), 'int')  # save the center coordinate of each nucleus

    slices = ndimage.find_objects(pred_result)
    for i, si in enumerate(slices):
        if si is not None:
            sr, sc = si
            yi, xi = np.nonzero(pred_result[sr, sc] == (i + 1))
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
    info.to_csv(save_path, sep=",", header=True, index=False)

    return info


# filter nuleus
def filter_nucleus(save_path, mode, top_value, bottom_value):
    cell_info = pd.read_csv(os.path.join(save_path, "cell_center.csv"), sep=",", header=0)
    if mode == 1:
        cell_info = cell_info[
            (cell_info["area"] > (bottom_value ** 2) * 3.14) & (cell_info["area"] < (top_value ** 2) * 3.14)]
    if mode == 2:
        cell_info.sort_values(by="area", inplace=True, ascending=True)
        cell_num = cell_info.shape[0]
        cell_info = cell_info.iloc[int(cell_num / 100 * bottom_value):cell_num - int(cell_num / 100 * top_value)]

    cell_info.to_csv(os.path.join(save_path, "filtered_cell_center.csv"), sep=",", header=True, index=False)


def nucleus_recongnition_cut_img(data_path, output_path, device, mode, top_value, bottom_value, threshold):
    device = "cuda" if device == "GPU" else "cpu"
    star_time = time.time()
    all_section = os.listdir(data_path)
    for item in all_section:
        dapi_path = find_dapi_img(os.path.join(data_path, item))
        save_path = os.path.join(output_path, item, "1_nucleus_recongnition_result")
        os.makedirs(save_path, exist_ok=True)

        # cut images
        cut_img_result_path = make_dir(save_path, "cut_img_result")
        cut_img(dapi_path, cut_img_result_path)

        # # Detected nucleus and determine the center of cell
        prediction_result_path = make_dir(save_path, "nucleus_detect_result")
        boundary_result_path = make_dir(save_path, "nucleus_boundary_result")
        predict(save_path, cut_img_result_path, prediction_result_path, boundary_result_path, device, threshold)

        # filter nuleus
        filter_nucleus(save_path, mode, top_value, bottom_value)

    print(f" detecting nucleus is finished, runtime：{time.time() - star_time}")
