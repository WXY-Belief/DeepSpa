import os
import open3d as o3d
import numpy as np
import pandas as pd
import copy
import random
import colorsys
import plotly
import plotly.graph_objects as go

size = 3


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
    plotly.offline.plot(fig, filename=save_path, auto_open=False)


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


def read_and_merge_data(path):
    all_section_cell_center_path = os.path.join(path, "all_section_result/cell_center.csv")
    all_section_cell_center = pd.read_csv(all_section_cell_center_path, sep=",", header=0)
    all_section_cell_center.drop("area", axis=1, inplace=True)

    all_section_cell_type_path = os.path.join(path, "all_section_result/cell_type.csv")
    all_section_cell_type = pd.read_csv(all_section_cell_type_path, sep=",", header=0)

    all_section_sec_anatomic_region_path = os.path.join(path,
                                                        "all_section_result/all_sec_anatomic_region_cluster_result.csv")

    all_section_sec_anatomic_region = pd.read_csv(all_section_sec_anatomic_region_path, sep=",", header=0)

    merge_data_1 = pd.merge(all_section_cell_center, all_section_cell_type, on=["section", "cell_index"])
    merge_data_2 = pd.merge(merge_data_1, all_section_sec_anatomic_region, on=["section", "cell_index"])
    merge_data_2.drop(["z", "angle"], axis=1, inplace=True)
    gap = 100
    all_section = sorted(merge_data_2["section"].drop_duplicates().tolist())
    all_z = pd.DataFrame()
    all_z["section"] = all_section
    all_z["z"] = np.array(list(range(1, len(all_section) + 1))) * gap

    final_point = pd.merge(merge_data_2, all_z, on="section")

    return final_point


def draw_all_cell_type(info, type_name, color_list, save_path):
    clouds = []
    for item in type_name:
        single_cell_type = info[info["cell_type"] == item]
        this_domain = o3d.core.Tensor(single_cell_type[["col", "row", "z"]].to_numpy())
        pcd_this_domain = o3d.t.geometry.PointCloud(this_domain)
        mat_this_domain = o3d.visualization.rendering.MaterialRecord()
        mat_this_domain.shader = 'defaultLitTransparency'
        this_color = list(np.array(color_list[type_name.index(item)]) / 255)
        this_color.append(1.0)
        mat_this_domain.base_color = this_color
        mat_this_domain.point_size = size
        clouds.append({'name': 'this_domain' + str(item), 'geometry': pcd_this_domain, 'material': mat_this_domain})
        print(clouds)
    o3d.visualization.draw(clouds, show_skybox=False, bg_color=[1.0, 1.0, 1.0, 1.0])
    draw_legned(type_name, color_list, os.path.join(save_path, "cell_type_legend.html"))


def draw_single_cell_type(info, type_name, color_list):
    for item in type_name:
        clouds = []
        single_cell_type = info[info["cell_type"] == item]
        rest_point = info.drop(single_cell_type.index, axis=0)
        this_domain = o3d.core.Tensor(single_cell_type[["col", "row", "z"]].to_numpy())
        pcd_this_domain = o3d.t.geometry.PointCloud(this_domain)
        mat_this_domain = o3d.visualization.rendering.MaterialRecord()
        mat_this_domain.shader = 'defaultLitTransparency'
        print(type_name.index(item))
        print(color_list[type_name.index(item)])
        print(np.array(color_list[type_name.index(item)]))
        this_color = list(np.array(color_list[type_name.index(item)]) / 255)
        this_color.append(1.0)
        mat_this_domain.base_color = this_color
        mat_this_domain.point_size = size

        rest_domain = o3d.core.Tensor(rest_point[["col", "row", "z"]].to_numpy())
        pcd_rest_domain = o3d.t.geometry.PointCloud(rest_domain)
        mat_rest_domain = o3d.visualization.rendering.MaterialRecord()
        mat_rest_domain.shader = 'defaultLitTransparency'
        mat_rest_domain.base_color = [0.5, 0.5, 0.5, 0.5]
        mat_rest_domain.point_size = size
        clouds.append({'name': 'this_domain_' + str(item), 'geometry': pcd_this_domain, 'material': mat_this_domain})
        clouds.append({'name': 'rest_domain_' + str(item), 'geometry': pcd_rest_domain, 'material': mat_rest_domain})
        o3d.visualization.draw(clouds, show_skybox=False, bg_color=[1.0, 1.0, 1.0, 1.0])


def draw_all_anatomic_region(info, type_name, color_list, save_path):
    clouds = []
    for item in type_name:
        single_cell_type = info[info["anatomic_region"] == item]
        this_domain = o3d.core.Tensor(single_cell_type[["col", "row", "z"]].to_numpy())
        pcd_this_domain = o3d.t.geometry.PointCloud(this_domain)
        mat_this_domain = o3d.visualization.rendering.MaterialRecord()
        mat_this_domain.shader = 'defaultLitTransparency'
        this_color = list(np.array(color_list[type_name.index(item)]) / 255)
        this_color.append(1.0)
        mat_this_domain.base_color = this_color
        mat_this_domain.point_size = size
        clouds.append({'name': 'this_domain' + str(item), 'geometry': pcd_this_domain, 'material': mat_this_domain})
        print(clouds)
    o3d.visualization.draw(clouds, show_skybox=False, bg_color=[1.0, 1.0, 1.0, 1.0])
    draw_legned(type_name, color_list, os.path.join(save_path, "anatomic_region_legend.html"))


def draw_single_anatomic_region(info, type_name, color_list):
    for item in type_name:
        clouds = []
        single_cell_type = info[info["anatomic_region"] == item]
        rest_point = info.drop(single_cell_type.index, axis=0)
        this_domain = o3d.core.Tensor(single_cell_type[["col", "row", "z"]].to_numpy())
        pcd_this_domain = o3d.t.geometry.PointCloud(this_domain)
        mat_this_domain = o3d.visualization.rendering.MaterialRecord()
        mat_this_domain.shader = 'defaultLitTransparency'
        this_color = list(np.array(color_list[type_name.index(item)]) / 255)
        this_color.append(1.0)
        mat_this_domain.base_color = this_color
        mat_this_domain.point_size = 2.0

        rest_domain = o3d.core.Tensor(rest_point[["col", "row", "z"]].to_numpy())
        pcd_rest_domain = o3d.t.geometry.PointCloud(rest_domain)
        mat_rest_domain = o3d.visualization.rendering.MaterialRecord()
        mat_rest_domain.shader = 'defaultLitTransparency'
        mat_rest_domain.base_color = [0.5, 0.5, 0.5, 0.5]
        mat_rest_domain.point_size = 2.0
        clouds.append({'name': 'this_domain_' + str(item), 'geometry': pcd_this_domain, 'material': mat_this_domain})
        clouds.append({'name': 'rest_domain_' + str(item), 'geometry': pcd_rest_domain, 'material': mat_rest_domain})
        o3d.visualization.draw(clouds, show_skybox=False, bg_color=[1.0, 1.0, 1.0, 1.0])


def draw_surface(data, surface_color, save_path):
    handle = open(os.path.join(save_path, "surface.pcd"), "w")
    handle.write(
        '# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1')
    handle.write('\nWIDTH ' + str(data.shape[0]))
    handle.write('\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0')
    handle.write('\nPOINTS ' + str(data.shape[0]))
    handle.write('\nDATA ascii')

    for idx, item in data.iterrows():  # 这里我只用到了前三列，故只需展示0，1，2三列 读者可根据自身需要写入其余列
        string = '\n' + str(item["col"]) + ' ' + str(item["row"]) + ' ' + str(item["z"])
        handle.write(string)
    handle.close()

    surface_pcd = o3d.io.read_point_cloud(os.path.join(save_path, "surface.pcd"))
    mesh1 = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(surface_pcd, alpha=80)
    mesh1.compute_vertex_normals()
    mesh1.paint_uniform_color(list(np.array(surface_color) / 255))  # 指定显示为绿色

    mesh_out = mesh1.filter_smooth_simple(number_of_iterations=10)
    mesh_out.compute_vertex_normals()
    mesh_out = mesh_out.subdivide_loop(number_of_iterations=2)
    o3d.visualization.draw({'name': 'surface', 'geometry': mesh_out}, show_skybox=False, bg_color=[1.0, 1.0, 1.0, 1.0])


def draw_3d(output_path, draw_type):
    save_path = os.path.join(output_path, "3D_point_cloud")
    os.makedirs(save_path, exist_ok=True)

    merge_data = read_and_merge_data(output_path)
    print(merge_data.head(3))
    if draw_type == "surface" or draw_type == "all":
        surface_color = generate_colors(1)[0]
        draw_surface(merge_data, surface_color, save_path)

    if draw_type == "cell_type" or draw_type == "all":
        cell_type = merge_data["cell_type"].drop_duplicates().tolist()
        cell_type_color_list = generate_colors(len(cell_type))
        draw_single_cell_type(merge_data, cell_type, cell_type_color_list)
        draw_all_cell_type(merge_data, cell_type, cell_type_color_list, save_path)

    if draw_type == "anatomic_region" or draw_type == "all":
        anatomic_region = merge_data["anatomic_region"].drop_duplicates().tolist()
        anatomic_region_color = generate_colors(len(anatomic_region))
        draw_single_anatomic_region(merge_data, anatomic_region, anatomic_region_color)
        draw_all_anatomic_region(merge_data, anatomic_region, anatomic_region_color, save_path)
