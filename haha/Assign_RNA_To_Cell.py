import os
import time
import numpy as np
import pandas as pd
import warnings
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")


def filter_unqualified_cell(cell_center_path, save_path, cell_express_min_number):
    cell_center = pd.read_csv(cell_center_path, sep=",", header=0)
    express_matrix = pd.read_csv(os.path.join(save_path, "GEM.csv"), sep=",", header=0, index_col=0)
    draw_need_file = pd.read_csv(os.path.join(save_path, "RNA_and_nearest_cell.csv"), sep=",", header=0)

    express_matrix.loc["express_sum"] = express_matrix.apply(lambda x: x.sum())

    drop_cell = express_matrix.columns[express_matrix.loc["express_sum"] < cell_express_min_number].tolist()
    print(f"The number of cells with a total expression of less than {cell_express_min_number} is：", drop_cell.shape[0])

    express_matrix.drop(drop_cell, axis=1, inplace=True)
    express_matrix.drop("express_sum", axis=0, inplace=True)
    express_matrix.to_csv(os.path.join(save_path, "filtered_GEM.csv"), sep=",", header=True, index=True)

    cell_center = cell_center[cell_center["cell_index"].isin(express_matrix.columns.tolist())]
    cell_center.to_csv(os.path.join(save_path, "filtered_cell_center_coordinate.csv"),
                       sep=",", header=True, index=False)

    draw_need_file = draw_need_file[draw_need_file["cell_index"].isin(express_matrix.columns.tolist())]
    draw_need_file.to_csv(os.path.join(save_path, "filtered_RNA_and_nearest_cell.csv"),
                          sep=",", header=True, index=False)

    return cell_center


def generate_gene_by_cell_matrix(rna_and_nearest_cell, gene_name, save_path):
    all_cell_index = rna_and_nearest_cell["cell_index"].drop_duplicates().tolist()

    infos = pd.DataFrame()
    for item in all_cell_index:
        info = dict(zip(gene_name, [0] * len(gene_name)))

        single_cell = rna_and_nearest_cell[rna_and_nearest_cell["cell_index"] == item]
        single_cell = single_cell.groupby('gene').count()
        info.update(dict(zip(single_cell.index.to_list(), single_cell["cell_index"].to_list())))

        infos[item] = np.array(list(info.values()))
    infos.index = gene_name
    infos.to_csv(os.path.join(save_path, "GEM.csv"), sep=",", header=True, index=True)


def assign_rna_to_cell(data_path, output_path, max_distance, flag, cell_express_min_number):
    star_time = time.time()
    all_section = os.listdir(data_path)

    all_cell_center = pd.DataFrame()
    for item in all_section:
        print(f"<-------------section:{item}--------------->")
        save_path = os.path.join(output_path, item, "3_gem")
        os.makedirs(save_path)
        if flag == 0:
            rna_coor_path = os.path.join(data_path, item, "rna_coordinate.csv")
            cell_center_path = os.path.join(output_path, item, "1_nucleus_recongnition_result/filtered_cell_center.csv")

        else:
            rna_coor_path = os.path.join(output_path, item, "2_Aligned_result/rotate_rna_coordinate.csv")
            cell_center_path = os.path.join(output_path, item, "2_Aligned_result/rotate_cell_center.csv")

        rna_coor = pd.read_csv(rna_coor_path, sep=",", header=0, index_col=None)
        cell_center = pd.read_csv(cell_center_path, sep=",", header=0, index_col=None)

        rna_coor_np = rna_coor[["row", "col"]].values
        cell_center_np = cell_center[["row", "col"]].values

        distances = cdist(rna_coor_np, cell_center_np)

        min_value = np.min(distances, axis=1)
        min_value_index = np.argmin(distances, axis=1)
        min_value_cell_index = cell_center.loc[min_value_index]["cell_index"].to_numpy()
        rna_coor["min_distace"] = min_value
        rna_coor["cell_index"] = min_value_cell_index

        generate_draw_file = rna_coor[rna_coor["min_distace"] < max_distance]
        generate_draw_file.to_csv(os.path.join(save_path, "draw_need_file.csv"), sep=",", header=True,
                                  index=False)

        gene_name = sorted(rna_coor["gene"].drop_duplicates().tolist())
        generate_gene_by_cell_matrix(generate_draw_file, gene_name, save_path)

        background_num = rna_coor.shape[0] - generate_draw_file.shape[0]
        print(f"the number of background RNA：{background_num}")

        cell_center = filter_unqualified_cell(cell_center, save_path, cell_express_min_number)
        cell_center["section"] = item
        all_cell_center = pd.concat([all_cell_center, cell_center])

    all_section_result_path = os.path.join(output_path, "all_section_result")
    os.makedirs(all_section_result_path, exist_ok=True)
    all_cell_center.to_csv(os.path.join(all_section_result_path, "cell_center.csv"), sep=",", header=True)

    print(f"generate expression matrix finished, runtime：{time.time() - star_time}")
