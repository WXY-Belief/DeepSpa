import os
import cv2
import numpy as np
import copy
from skimage.segmentation import find_boundaries
import mmcv
from tqdm import tqdm
from tools.test import update_legacy_cfg

from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.datasets import result_to_inst

config = "configs/dsb_mitb2_cellpose.py"  # 这个是配置文件，一般不用改
checkpoint = "final_model_tissuenet/iter_52500.pth"  # 权重路径，一共有两个，
# final_model/iter_56000.pth
# 和
# final_model_tissuenet/iter_52500.pth
img = "DAPI.tif"  # 要预测的图片路径
device = "cuda"  # 显卡就是"cuda"，cpu就是"cpu"
# build the model from a config file and a checkpoint file
cfg = mmcv.Config.fromfile(config)
# cfg = update_legacy_cfg(cfg)
model = init_segmentor(
    cfg,
    checkpoint,
    device=device,
    revise_checkpoint=[(r'^module\.', ''), ('model.', '')])

img_dir = "split_20/"
imgs = []
preds = []
for img in tqdm(os.listdir(img_dir)):
    result = inference_segmentor(model, img_dir + img)
    pred = result_to_inst(result[0])[0]  # 这个是最后的结果
    imgs.append(img)
    preds.append(pred)


def make_outline_overlay(RGB_data, predictions):
    boundaries = np.zeros_like(predictions)
    overlay_data = copy.copy(RGB_data)

    for img in range(predictions.shape[0]):
        boundary = find_boundaries(predictions[img, :, :], connectivity=1, mode='inner')
        boundaries[img, boundary > 0] = 1

    overlay_data[boundaries > 0, :] = (0, 200, 0)

    return overlay_data


os.makedirs("split_20_pred", exist_ok=True)
for idx, img_file in enumerate(imgs):
    img = cv2.imread("split/" + img_file)
    pred = preds[idx]
    overlay = make_outline_overlay(
        np.expand_dims(img, 0),
        np.expand_dims(pred, 0)
    )

    cv2.imwrite("split_20_pred/" + img_file, overlay[0])
