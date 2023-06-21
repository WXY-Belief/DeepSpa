import os
import time

import pandas as pd
import plotly
import plotly.graph_objects as go
import random
import colorsys


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


def draw_all(all_point, save_path, categroy, max_z):
    fig = go.Figure()

    for item in categroy:
        x = all_point[all_point["categroy"] == item]["col"]
        y = all_point[all_point["categroy"] == item]["row"]
        z = all_point[all_point["categroy"] == item]["z"]
        fig.add_scatter3d(x=x,
                          y=y,
                          z=z,
                          mode='markers',
                          name=item,
                          marker=dict(size=2, color=all_point[all_point["categroy"] == item]["color"]),
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
            zaxis=dict(nticks=4, range=[0, max_z + 1])),

        margin=dict(r=20, l=10, b=10, t=10))
    fig.update_layout(showlegend=True,
                      legend=dict(yanchor="top", y=0.79, xanchor="left", x=0.01, font=dict(size=10))
                      )
    fig.update_layout(margin=dict(l=100, r=100, b=100, t=100))
    plotly.offline.plot(fig, filename=save_path, auto_open=False)


def draw_single(rest_categray_cell, single_caegray, categroy, draw_3d_result_path, max_z):
    fig = go.Figure(data=[go.Scatter3d(x=rest_categray_cell['col'],
                                       y=rest_categray_cell['row'],
                                       z=rest_categray_cell['z'],
                                       mode='markers',
                                       name="rest_categroy",
                                       marker=dict(size=2, color="rgb(114,114,114)"),
                                       opacity=0.05,
                                       connectgaps=True),
                          go.Scatter3d(x=single_caegray['col'],
                                       y=single_caegray['row'],
                                       z=single_caegray['z'],
                                       mode='markers',
                                       name=str(categroy),
                                       marker=dict(size=2, color=single_caegray["color"]),
                                       connectgaps=True)
                          ],
                    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        template=None)
    fig.update_layout(
        scene=dict(
            zaxis=dict(nticks=4, range=[0, max_z])),

        margin=dict(r=20, l=10, b=10, t=10))
    fig.update_layout(showlegend=True,
                      legend=dict(yanchor="top", y=0.79, xanchor="left", x=0.01, font=dict(size=10))
                      )
    fig.update_layout(margin=dict(l=100, r=100, b=100, t=100))
    plotly.offline.plot(fig, filename=draw_3d_result_path, auto_open=False)


def draw_3d(all_section_cell_center, all_section_cell_type, all_section_anatomic_region, output_path):
    start_time = time.time()
    base_dir = os.path.join(output_path, "3D_result")

    cell_type_save_path = os.path.join(base_dir, "cell_type_3D")
    os.makedirs(cell_type_save_path, exist_ok=True)

    all_section_cell_center.drop("area", axis=1, inplace=True)
    all_section_z = pd.DataFrame()
    all_section_z["section"] = all_section_cell_center["section"].drop_duplicates().tolist()
    all_section_z["z"] = list(range(1, all_section_z.shape[0] + 1))
    all_section_cell_center = pd.merge(all_section_cell_center, all_section_z, on="section")

    cell_type_3d = pd.merge(all_section_cell_center, all_section_cell_type, on=["section", "cell_type"])
    cell_type_name = all_section_cell_type["cell_type"].drop_duplicates().tolist()
    cell_type_color = generate_colors(len(cell_type_name))
    all_categroy_color = pd.DataFrame()
    all_categroy_color["cell_type"] = cell_type_name
    all_categroy_color["color"] = cell_type_color
    cell_type_3d = pd.merge(cell_type_3d, all_categroy_color, on="cell_type")
    cell_type_3d.rename({"cell_type": "categroy"}, axis=1, inplace=True)

    draw_all(cell_type_3d, os.path.join(cell_type_save_path, "all_cell_type.html"), cell_type_name,
             all_section_z.shape[0] + 1)

    single_cell_type_save_path = os.path.join(cell_type_save_path, "single_cell_type")
    os.makedirs(single_cell_type_save_path, exist_ok=True)

    for item in cell_type_name:
        single_categroy = cell_type_3d[cell_type_3d["categroy"] == item]
        rest_categray_cell = cell_type_3d[(True ^ cell_type_3d['categroy'].isin([item]))]
        draw_3d_result_path = os.path.join(single_cell_type_save_path, str(item) + ".html")
        draw_single(rest_categray_cell, single_categroy, item, draw_3d_result_path, all_section_z.shape[0] + 1)

    # anatomic region 3D
    anatomic_region_save_path = os.path.join(base_dir, "anatomic_region_3D")
    os.makedirs(anatomic_region_save_path, exist_ok=True)

    anatomic_region_3d = pd.merge(all_section_cell_center, all_section_anatomic_region, on=["section", "anatomic_region"])
    anatomic_region_name = all_section_anatomic_region["anatomic_region"].drop_duplicates().tolist()
    anatomic_region_color = generate_colors(len(anatomic_region_name))

    all_categroy_color = pd.DataFrame()
    all_categroy_color["anatomic_region"] = anatomic_region_name
    all_categroy_color["color"] = anatomic_region_color
    anatomic_region_3d = pd.merge(anatomic_region_3d, all_categroy_color, on="anatomic_region")
    anatomic_region_3d.rename({"anatomic_region": "categroy"}, axis=1, inplace=True)

    draw_all(anatomic_region_3d, os.path.join(anatomic_region_save_path, "all_anatomic_region.html"), anatomic_region_name,
             all_section_z.shape[0] + 1)

    single_anatomic_region_save_path = os.path.join(anatomic_region_save_path, "single_anatomic_region")
    os.makedirs(single_anatomic_region_save_path, exist_ok=True)

    for item in anatomic_region_name:
        single_categroy = anatomic_region_3d[anatomic_region_3d["categroy"] == item]
        rest_categray_cell = anatomic_region_3d[(True ^ anatomic_region_3d['categroy'].isin([item]))]
        draw_3d_result_path = os.path.join(single_anatomic_region_save_path, str(item) + ".html")
        draw_single(rest_categray_cell, single_categroy, item, draw_3d_result_path, all_section_z.shape[0] + 1)

    print("Drawing 3D result finished, runtime：", time.time() - start_time)