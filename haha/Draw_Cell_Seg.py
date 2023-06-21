import os.path
import time
import pandas as pd
import numpy as np
import cv2
import random
import seaborn as sns
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def draw_cell_seg(data_path, output_path, flag):
    star_time = time.time()
    all_section = os.listdir(data_path)
    for item in all_section:
        save_path = os.path.join(output_path, item, "4_cell_segmentation_image")
        os.makedirs(save_path)
        cell_center = pd.read_csv(os.path.join(output_path, item, "filtered_cell_center_coordinate.csv"), sep=",",
                                  header=0)
        cell_and_rna = pd.read_csv(os.path.join(output_path, item, "filtered_RNA_and_nearest_cell.csv"), sep=",",
                                   header=0)

        if flag == 0:
            dapi = Image.open(os.path.join(data_path, item, "DAPI.tif"))
        else:
            dapi = Image.open(os.path.join(output_path, item, "2_Aligned_result/align_img.PNG"))

        img = np.zeros((dapi.height, dapi.width, 3), dtype=np.uint8)
        img[:, :, :] = 255

        color_list = np.array(sns.color_palette('tab20', 10)) * 255
        for idx, item_1 in cell_center.iterrows():
            this_cell_rna = cell_and_rna[cell_and_rna["cell_index"] == item_1["cell_index"]]
            color = tuple(color_list[random.randrange(0, 10)])
            cv2.circle(img=img, center=(int(item_1["col"]), int(item_1["row"])), radius=20,
                       color=color, thickness=-1)

            for idx_1, item_2 in this_cell_rna.iterrows():
                cv2.circle(img=img, center=(int(item_2["col"]), int(item_2["row"])), radius=10,
                           color=color, thickness=-1)
                cv2.line(img=img, pt1=(int(item_1["col"]), int(item_1["row"])),
                         pt2=(int(item_2["col"]), int(item_2["row"])), color=color, thickness=3, lineType=cv2.LINE_4)

        cv2.imwrite(os.path.join(save_path, "cell_segmentation.PNG"), img)

    print(f"Drawing image of cell segmentation finished, runtime:{time.time() - star_time}")
