import os.path
import time
import colorsys
import random
import numpy as np
import cv2
import warnings
import pandas as pd
import plotly
import plotly.graph_objects as go
import copy
from PIL import Image

warnings.filterwarnings("ignore")


def draw_legned(categroy, legend_color, save_path):
    color = copy.copy(legend_color)
    for i in range(0, len(color)):
        temp = np.array(color[i])[::-1]
        color[i] = "rgb" + str(tuple([int(temp[0]), int(temp[1]), int(temp[2])]))
    length = len(color)

    fig = go.Figure()
    for i in range(0, length):
        x = [i + 1]
        y = [i + 1]
        z = [i + 1]
        fig.add_scatter3d(x=x,
                          y=y,
                          z=z,
                          mode='markers',
                          name=categroy[i],
                          marker=dict(size=5, color=color[i]),
                          connectgaps=True)
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        template=None)
    fig.update_layout(
        scene=dict(
            zaxis=dict(nticks=4, range=[0, len(color) + 1])),

        margin=dict(r=20, l=10, b=10, t=10))
    fig.update_layout(showlegend=True,
                      legend=dict(yanchor="top", y=0.79, xanchor="left", x=0.2, font=dict(size=10))
                      )
    fig.update_layout(margin=dict(l=100, r=100, b=100, t=100))
    plotly.offline.plot(fig, filename=os.path.join(save_path, "legend.html"), auto_open=False)


def generate_colors(num_colors):
    colors = []
    random.seed(123)  # 设置随机种子
    for i in range(num_colors):
        hue = random.random()  # 随机选择色调值
        saturation = random.uniform(0.5, 1.0)  # 随机选择饱和度值（范围可调整）
        value = random.uniform(0.5, 1.0)  # 随机选择亮度值（范围可调整）
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        rgb_int = [int(c * 255) for c in rgb]
        colors.append(rgb_int)
    return colors


def draw_anatomic_region_map(all_section_cell_center, all_section_anatomic_region, data_path, output_path, flag):
    star_time = time.time()

    all_section = all_section_cell_center["section"].drop_duplicates().tolist()
    anatomic_region_name = all_section_anatomic_region["anatomic_region"].drop_duplicates().tolist()
    color_list = generate_colors(len(anatomic_region_name))

    for item in all_section:
        save_path = os.path.join(output_path, str(item), "7_anatomic_region_result")
        rna_and_nearest_cell_path = os.path.join(output_path, str(item), "3_gem", "filtered_RNA_and_nearest_cell.csv")
        rna_and_nearest_cell = pd.read_csv(rna_and_nearest_cell_path, sep=",", header=0)

        cell_center = all_section_cell_center[all_section_cell_center["section"] == item]
        anatomic_region = all_section_anatomic_region[all_section_anatomic_region["section"] == item]

        if flag == 0:
            DAPI = Image.open(os.path.join(data_path, str(item), "DAPI.tif"))
        else:
            DAPI = Image.open(os.path.join(output_path, str(item), "2_Aligned_result", "align_img.PNG"))

        merge_data = pd.merge(anatomic_region, cell_center, on="cell_index")
        img = np.zeros((DAPI.height, DAPI.width, 3), dtype=np.uint8)
        img[:, :, :] = 255

        for idx_1, item_1 in merge_data.iterrows():
            cell_and_rna = rna_and_nearest_cell[rna_and_nearest_cell["cell_index"] == item_1["cell_index"]]
            color = color_list[anatomic_region_name.index(item_1["anatomic_region"])]
            cv2.circle(img=img, center=(int(item_1["col"]), int(item_1["row"])), radius=20,
                       color=color, thickness=-1)

            for idx_2, item_2 in cell_and_rna.iterrows():
                cv2.circle(img=img, center=(int(item_2["col"]), int(item_2["row"])), radius=10,
                           color=color, thickness=-1)
                cv2.line(img=img, pt1=(int(item_1["col"]), int(item_1["row"])),
                         pt2=(int(item_2["col"]), int(item_2["row"])), color=color, thickness=3, lineType=cv2.LINE_4)

        cv2.imwrite(os.path.join(save_path, "anatomic_region.PNG"), img)

        draw_legned(anatomic_region_name, color_list, save_path)
    print("Drawing image of anatomic region finished, runtime：", time.time() - star_time)
