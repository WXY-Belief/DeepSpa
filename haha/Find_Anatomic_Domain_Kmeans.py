import os
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


def kmeans_clustering(distri_matrix, clustering_result_path, cluster_num):
    clustering_result = pd.DataFrame()
    clustering_result["cell_index"] = distri_matrix["cell_index"]
    clustering_result["section"] = distri_matrix["section"]

    distri_matrix.drop(["cell_index", "section"], axis=1, inplace=True)
    cluster = KMeans(n_clusters=cluster_num, random_state=0).fit(distri_matrix)

    clustering_result["anatomic_region"] = cluster.labels_

    clustering_result.to_csv(clustering_result_path, sep=",", header=True, index=False)

    return clustering_result


def find_anatomic_domain(all_section_cell_center, all_section_cell_type, output_path, neighbor_num,
                         anatomic_domain_num):
    start_time = time.time()
    all_sec_cell_k_neighbor = pd.DataFrame()
    all_section = all_section_cell_center["section"].drop_duplicates().tolist()
    cell_type_name = all_section_cell_type["cell_type"].drop_duplicates().tolist()
    for item in all_section:
        save_path = os.path.join(output_path, str(item), "7_anatomic_region_result")
        cell_center = all_section_cell_center[all_section_cell_center["section"] == item]
        cell_center.drop("section", axis=1, inplace=True)

        cell_type = all_section_cell_type[all_section_cell_type["section"] == item]
        cell_type.drop("section", axis=1, inplace=True)

        merge_data = pd.merge(cell_type, cell_center, on="cell_index")

        merge_data_cell_coor = merge_data[["row", "col"]].values
        distance = cdist(merge_data_cell_coor, merge_data_cell_coor)

        min_k_distance_index = np.argpartition(distance, neighbor_num, axis=1)[:, :neighbor_num]
        # min_k_distance = np.take_along_axis(distance, min_k_distance_index, axis=1)

        list_cell_index = merge_data["cell_index"].tolist()
        distribution = pd.DataFrame()
        for i, row in enumerate(min_k_distance_index):
            info = dict(zip(cell_type_name, [0] * len(cell_type_name)))
            k_cell = merge_data.loc[row].groupby(by="cell_type").count()

            info.update(dict(zip(k_cell.index.to_list(), k_cell["cell_index"].to_list())))
            distribution[list_cell_index[i]] = list(info.values())

        distribution.loc["section"] = [item] * cell_center.shape[0]
        distribution.index = cell_type_name

        all_sec_cell_k_neighbor = pd.concat([all_sec_cell_k_neighbor, distribution.T])
        distribution = distribution.T
        distribution["index"] = distribution.index.to_list()
        distribution.drop("section", axis=1, inplace=True)
        distribution.to_csv(os.path.join(save_path, "cell_K_neighbor.csv"), sep=",", header=True, index=False)

    all_sec_cell_k_neighbor["cell_index"] = all_sec_cell_k_neighbor.index.to_list()
    all_sec_cell_k_neighbor.to_csv(os.path.join(output_path, "all_section_result/all_sec_cell_K_neighbor.csv"),
                                   sep=",", header=True, index=False)

    all_sec_clustering_result_path = os.path.join(output_path,
                                                  "all_section_result/all_sec_anatomic_region_cluster_result.csv")
    result = kmeans_clustering(all_sec_cell_k_neighbor, all_sec_clustering_result_path, anatomic_domain_num)

    for item in all_section:
        save_path = os.path.join(output_path, str(item), "7_anatomic_region_result")
        single_result = result[result["section"] == item]
        single_result.drop("section", axis=1, inplace=True)
        single_result.to_csv(os.path.join(save_path, "anatomic_region.csv"), sep=",", header=True, index=False)

    print("Find anatomic regions finished, runtime：", time.time() - start_time)

    return result
