import os
import time
import pandas as pd
import toml
from haha.Draw_Cell_Map import draw_cell_type_map
from haha.Find_Anatomic_Domain_Kmeans import find_anatomic_domain
from haha.Draw_Anatomic_Region import draw_anatomic_region_map
from haha.Draw_3D import draw_3d

if __name__ == "__main__":
    star_time = time.time()

    # Load configuration file
    configration_file_path = "./Configration.toml"
    all_parameter = toml.load(configration_file_path)

    data_path = all_parameter["data_path"]  # Directory where the data is located
    output_path = all_parameter["output_path"]  # Directory where the output results are saved
    device = all_parameter["device"]
    flag = all_parameter["section_align_flag"]

    all_section_cell_center = pd.read_csv(os.path.join(output_path, "all_section_result/cell_center.csv"), sep=",",
                                          header=0)
    all_section_cell_type = pd.read_csv(os.path.join(output_path, "all_section_result/cell_type.csv"), sep=",",
                                        header=0)

    # 6.draw cell type map result
    draw_cell_type_map(all_section_cell_center, all_section_cell_type, data_path, output_path, flag)

    # 7.find anatomic regions
    K = all_parameter["K"]
    anatomic_domain_num = all_parameter["anatomic_domain_num"]
    all_section_anatomic_region = find_anatomic_domain(all_section_cell_center, all_section_cell_type, output_path, K,
                                                       anatomic_domain_num)

    # 7_1.draw anatomic regions
    draw_anatomic_region_map(all_section_cell_center, all_section_anatomic_region, data_path, output_path, flag)

    # 8. draw 3D of cell types and anatomic regions
    draw_3d_mode = all_parameter["draw_3d_mode"]
    if draw_3d_mode == 1:
        draw_3d(all_section_cell_center, all_section_cell_type, all_section_anatomic_region, output_path)
    else:
        pass
