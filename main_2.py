import os
import argparse
import pandas as pd
import toml
from haha.Cell_Type_Map import cell_type_map
from haha.Draw_Cell_Map import draw_cell_type_map
from haha.Find_Anatomic_Domain_Kmeans import find_anatomic_domain
from haha.Draw_Anatomic_Region import draw_anatomic_region_map


def main(config_path):
    # Load configuration file
    all_parameter = toml.load(config_path)

    data_path = all_parameter["data_path"]  # Directory where the data is located
    output_path = all_parameter["output_path"]  # Directory where the output results are saved
    device = all_parameter["device"]
    flag = all_parameter["section_align_flag"]

    # 5.Cell type map
    cell_type_annotation_mode = all_parameter["cell_type_annotation_mode"]
    sc_data_path = all_parameter["sc_data_path"]
    if cell_type_annotation_mode == 1:
        cell_type_map(data_path, output_path, sc_data_path, device)
    else:
        all_section = os.listdir(data_path)
        for item in all_section:
            os.makedirs(os.path.join(output_path, item, "5_cell_type_result"), exist_ok=True)

    # merge result of all section
    all_section_cell_center = pd.DataFrame()
    all_section_cell_type = pd.DataFrame()
    all_result_save_path = os.path.join(output_path, "all_section_result")
    for item in os.listdir(data_path):
        cell_center_path = os.path.join(output_path, item, "3_gem", "filtered_cell_center_coordinate.csv")
        cell_type_path = os.path.join(output_path, item, "5_cell_type_result", "cell_type.csv")

        cell_center = pd.read_csv(cell_center_path, sep=",", header=0)
        cell_center["section"] = int(item)

        cell_type = pd.read_csv(cell_type_path, sep=",", header=0)
        cell_type["section"] = int(item)

        all_section_cell_center = pd.concat([all_section_cell_center, cell_center])
        all_section_cell_type = pd.concat([all_section_cell_type, cell_type])

        all_section_cell_center.to_csv(os.path.join(all_result_save_path, "cell_center.csv"), sep=",", header=True,
                                       index=False)

        all_section_cell_type.to_csv(os.path.join(all_result_save_path, "cell_type.csv"), sep=",", header=True,
                                     index=False)

    # 5_1.Draw cell type map
    draw_cell_type_map(all_section_cell_center, all_section_cell_type, data_path, output_path, flag)

    # 7.find anatomic regions
    K = all_parameter["K"]
    anatomic_domain_num = all_parameter["anatomic_region_num"]
    find_anatomic_domain(all_section_cell_center, all_section_cell_type, output_path, K,
                         anatomic_domain_num)

    # # 7_1.draw anatomic regions
    all_section_anatomic_region = pd.read_csv(
        os.path.join(output_path, "all_section_result", "all_sec_anatomic_region_cluster_result.csv"), sep=",", header=0)
    draw_anatomic_region_map(all_section_cell_center, all_section_anatomic_region, data_path, output_path, flag)


if __name__ == "__main__":
    # input parameter
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=str, default="Configration.toml")
    args = parser.parse_args()
    main(args.c)
