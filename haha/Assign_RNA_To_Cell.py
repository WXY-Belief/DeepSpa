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
        if mode == "stereo-seq":
            single_cell = single_cell.groupby('gene')["num"].count()
        if mode == "ISS":
            single_cell = single_cell.groupby('gene').count()
        info.update(dict(zip(single_cell.index.to_list(), single_cell["cell_index"].to_list())))

        infos[item] = np.array(list(info.values()))
    infos.index = gene_name
    infos.to_csv(os.path.join(save_path, "GEM.csv"), sep=",", header=True, index=True)


def iss_assign_rna_to_cell(cell_center, rna_coordinate, max_distance, mode, save_path):
    rna_coor_np = rna_coordinate[["row", "col"]].values
    cell_center_np = cell_center[["row", "col"]].values

    distances = cdist(rna_coor_np, cell_center_np)

    min_value = np.min(distances, axis=1)
    min_value_index = np.argmin(distances, axis=1)
    min_value_cell_index = cell_center.loc[min_value_index]["cell_index"].to_numpy()
    rna_coordinate["min_distance"] = min_value
    rna_coordinate["cell_index"] = min_value_cell_index

    generate_draw_file = rna_coordinate[rna_coordinate["min_distance"] < max_distance]
    generate_draw_file.to_csv(os.path.join(save_path, "RNA_and_nearest_cell.csv"), sep=",", header=True,
                              index=False)

    gene_name = sorted(rna_coordinate["gene"].drop_duplicates().tolist())
    generate_gene_by_cell_matrix(generate_draw_file, gene_name, mode, save_path)


def stereo_seq_assign_rna_to_cell(cell_center, rna_coordinate, max_distance, device, mode, save_path):
    if device == "GPU":
        device = torch.device("cuda:0")
        loop = 10000
    else:
        torch.device("cpu")
        loop = 100000

    gene_name = rna_coordinate["gene"].drop_duplicates().tolist()
    gpu_cell_center = torch.tensor(cell_center[["row", "col"]].to_numpy()).to(device)
    gpu_rna_coordinate = torch.tensor(rna_coordinate[["row", "col"]].to_numpy()).to(device)

    n = math.floor(rna_coordinate.shape[0] / loop)
    min_distances, min_distance_cell_indexs = np.array([]), np.array([])

    for i in range(loop + 1):
        if i == loop:
            this_gpu_rna_coordinate = gpu_rna_coordinate[i * n:, :]
        else:
            this_gpu_rna_coordinate = gpu_rna_coordinate[i * n: (i + 1) * n, :]

        diff = this_gpu_rna_coordinate[:, None] - gpu_cell_center[None, :]
        del this_gpu_rna_coordinate
        torch.cuda.empty_cache()
        distances = torch.sqrt(torch.sum(diff ** 2, dim=2))

        min_distance, min_distance_index = torch.min(distances, dim=1)
        del distances

        cpu_min_distance = min_distance.cpu().numpy()
        cpu_min_distance_index = min_distance_index.cpu().numpy()
        cell_index = cell_center.loc[cpu_min_distance_index, :]["cell_index"].to_numpy()

        min_distances = np.concatenate((min_distances, cpu_min_distance), axis=0)
        min_distance_cell_indexs = np.concatenate((min_distance_cell_indexs, cell_index), axis=0)

    rna_coordinate["min_distance"] = min_distances
    rna_coordinate["cell_index"] = min_distance_cell_indexs

    filter_rna_coordinate = rna_coordinate[rna_coordinate["min_distance"] < max_distance]
    filter_rna_coordinate.to_csv(os.path.join(save_path, "RNA_and_nearest_cell.csv"), sep=",", header=True,
                                 index=False)

    # generate gene-by-cell matrix
    generate_gene_by_cell_matrix(filter_rna_coordinate, gene_name, mode, save_path)


def assign_rna_to_cell(data_path, output_path, max_distance, flag, cell_express_min_number, cell_express_max_number,
                       mode, device):
    star_time = time.time()
    all_section = os.listdir(data_path)

    all_cell_center = pd.DataFrame()
    for item in all_section:
        print(f"<-------------section:{item}--------------->")
        save_path = os.path.join(output_path, item, "3_gem")
        os.makedirs(save_path, exist_ok=True)

        if flag == 0:
            rna_coor_path = os.path.join(data_path, item, "rna_coordinate.csv")
            cell_center_path = os.path.join(output_path, item, "1_nucleus_recongnition_result",
                                            "filtered_cell_center.csv")
        else:
            rna_coor_path = os.path.join(output_path, item, "2_Aligned_result", "rotate_rna_coordinate.csv")
            cell_center_path = os.path.join(output_path, item, "2_Aligned_result", "rotate_cell_center.csv")

        rna_coordinate = pd.read_csv(rna_coor_path, sep=",", header=0, index_col=None)
        cell_center = pd.read_csv(cell_center_path, sep=",", header=0, index_col=None)

        if mode == "ISS":
            iss_assign_rna_to_cell(cell_center, rna_coordinate, max_distance, mode, save_path)
        else:
            stereo_seq_assign_rna_to_cell(cell_center, rna_coordinate, max_distance, device, mode, save_path)

        cell_center, filter_rna_num = filter_unqualified_cell(cell_center, save_path, cell_express_min_number,
                                                              cell_express_max_number)
        cell_center["section"] = item
        all_cell_center = pd.concat([all_cell_center, cell_center])
        print(f"the number of background RNA ：{rna_coordinate.shape[0] - filter_rna_num}")

    print(f"generate expression matrix finished, runtime：{time.time() - star_time}")
