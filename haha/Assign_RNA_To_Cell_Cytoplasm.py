import os
import time
import torch
import numpy as np
import pandas as pd
import warnings
import math
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")


def filter_unqualified_cell(cell_center, save_path, cell_express_min_number, cell_express_max_number):
    express_matrix = pd.read_csv(os.path.join(save_path, "GEM.csv"), sep=",", header=0, index_col=0).T
    draw_need_file = pd.read_csv(os.path.join(save_path, "RNA_and_nearest_cell.csv"), sep=",", header=0)

    express_matrix["express_sum"] = express_matrix.sum(axis=1)

    if cell_express_max_number == 0:
        drop_cell = express_matrix[express_matrix["express_sum"] < cell_express_min_number].index.tolist()
    else:
        drop_cell = express_matrix[express_matrix["express_sum"] < cell_express_min_number or express_matrix[
            "express_sum"] > cell_express_max_number].index.tolist()
    print(f"The number of cells with a total expression of less than {cell_express_min_number} is：", len(drop_cell))

    express_matrix.drop(drop_cell, axis=0, inplace=True)
    express_matrix.drop("express_sum", axis=1, inplace=True)
    express_matrix.T.to_csv(os.path.join(save_path, "filtered_GEM.csv"), sep=",", header=True, index=True)

    rest_cell = [int(item) for item in express_matrix.index.tolist()]
    cell_center = cell_center[cell_center["cell_index"].isin(rest_cell)]
    cell_center.to_csv(os.path.join(save_path, "filtered_cell_center_coordinate.csv"),
                       sep=",", header=True, index=False)

    draw_need_file = draw_need_file[draw_need_file["cell_index"].isin(rest_cell)]
    draw_need_file.to_csv(os.path.join(save_path, "filtered_RNA_and_nearest_cell.csv"),
                          sep=",", header=True, index=False)

    return cell_center, draw_need_file.shape[0]


def generate_gene_by_cell_matrix(rna_and_nearest_cell, gene_name, mode, save_path):
    all_cell_index = rna_and_nearest_cell["cell_index"].drop_duplicates().tolist()

    infos = pd.DataFrame()
    for item in all_cell_index:
        info = dict(zip(gene_name, [0] * len(gene_name)))

        single_cell = rna_and_nearest_cell[rna_and_nearest_cell["cell_index"] == item]
        if mode == "NGS":
            single_cell = single_cell.groupby('gene')["num"].count()
        if mode == "ISS":
            single_cell = single_cell.groupby('gene').count()
        info.update(dict(zip(single_cell.index.to_list(), single_cell["cell_index"].to_list())))

        infos[item] = np.array(list(info.values()))
    infos.index = gene_name
    infos.to_csv(os.path.join(save_path, "GEM.csv"), sep=",", header=True, index=True)


def assign_rna_to_cell(data_path, output_path, cell_express_min_number, cell_express_max_number,
                       mode):
    star_time = time.time()
    all_section = os.listdir(data_path)

    all_cell_center = pd.DataFrame()
    for item in all_section:
        print(f"<-------------section:{item}--------------->")
        save_path = os.path.join(output_path, item, "2_gem")
        os.makedirs(save_path, exist_ok=True)

        rna_coor_path = os.path.join(data_path, item, "rna_coordinate.csv")
        cytoplasm_area_path = os.path.join(output_path, item, "1_nucleus_recongnition_result", "cytoplasm_area.npy")
        cell_center_path = os.path.join(output_path, item, "1_nucleus_recongnition_result",
                                        "filtered_cell_center.csv")
        cell_center = pd.read_csv(cell_center_path, sep=",", header=0, index_col=None)

        rna_coordinate = pd.read_csv(rna_coor_path, sep=",", header=0, index_col=None)
        cytoplasm_area = np.load(cytoplasm_area_path)

        RNA_to_cell = cytoplasm_area[rna_coordinate["row"].tolist(), rna_coordinate["col"].tolist()]

        rna_coordinate["cell_index"] = RNA_to_cell
        drop_index = rna_coordinate[rna_coordinate["cell_index"] == 0].index
        rna_coordinate.drop(drop_index, axis=0, inplace=True)

        gene_name = sorted(rna_coordinate["gene"].drop_duplicates().tolist())

        if mode == "ISS":
            generate_gene_by_cell_matrix(rna_coordinate, gene_name, "ISS", save_path)
        else:
            generate_gene_by_cell_matrix(rna_coordinate, gene_name, "NGS", save_path)

        cell_center, filter_rna_num = filter_unqualified_cell(cell_center, save_path, cell_express_min_number,
                                                              cell_express_max_number)

        cell_center["section"] = item
        all_cell_center = pd.concat([all_cell_center, cell_center])
        print(f"the number of background RNA ：{rna_coordinate.shape[0] - filter_rna_num}")

    print(f"generate expression matrix finished, runtime：{time.time() - star_time}")
