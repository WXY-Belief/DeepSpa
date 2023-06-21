import os
import pandas as pd
import plotly
import plotly.graph_objects as go


# 输入种类数，返回相同数量的颜色
def get_color(class_num):
    # 每个通道被分成几份
    x = 1
    while True:
        if x ** 3 > class_num:
            break
        else:
            x += 1
    # 每个通道中的两个值之间的差距
    distance = int(255 / x)

    # 最终返回的颜色列表
    barcode_color = []

    for i in range(0, x):
        for j in range(0, x):
            for k in range(0, x):
                candidate_color = "rgb" + str((i * distance, j * distance, k * distance))
                barcode_color.append(candidate_color)
    print(barcode_color)
    distance_num = int(len(barcode_color) / class_num)
    final_color = []
    # 只取clas_num个颜色
    for i in range(0, class_num):
        final_color.append(barcode_color[1 + i * distance_num])

    return final_color


def draw_all(all_point, save_path, categroy, max_z):
    save_path = os.path.join(save_path, "all_categroy_3d.html")
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
                                       name="rest_tissue_domain",
                                       marker=dict(size=2, color="rgb(114,114,114)"),
                                       opacity=0.05,
                                       connectgaps=True),
                          go.Scatter3d(x=single_caegray['col'],
                                       y=single_caegray['row'],
                                       z=single_caegray['z'],
                                       mode='markers',
                                       name="tissue_domain_" + str(categroy),
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


def draw_tissue_3d(draw_info, draw_3d_save_path):
    # 给不同切片一个Z轴
    draw_info.drop("area", axis=1, inplace=True)

    all_section = draw_info["section"].drop_duplicates().tolist()
    all_section_z = pd.DataFrame()
    all_section_z["section"] = all_section
    all_section_z["z"] = list(range(1, len(all_section) + 1))

    draw_info = pd.merge(draw_info, all_section_z, on="section")

    # 给所有细胞一个颜色
    all_categroy = draw_info["categroy"].drop_duplicates().tolist()
    color = get_color(len(all_categroy))
    all_categroy_color = pd.DataFrame()
    all_categroy_color["categroy"] = all_categroy
    all_categroy_color["color"] = color

    draw_info = pd.merge(draw_info, all_categroy_color, on="categroy")

    # 先画包含所有种类的
    draw_all(draw_info, draw_3d_save_path, all_categroy, len(all_section) + 1)

    # 画单个种类的
    for item_1 in all_categroy:
        single_caegray = draw_info[draw_info["categroy"] == item_1]
        rest_categray_cell = draw_info[(True ^ draw_info['categroy'].isin([item_1]))]
        draw_3d_result_path = os.path.join(draw_3d_save_path, "tissue_domain_" + str(item_1) + ".html")
        draw_single(rest_categray_cell, single_caegray, item_1, draw_3d_result_path, len(all_section) + 1)
