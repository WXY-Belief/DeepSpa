import os
import time
import toml
import pandas as pd
from haha.Nuclues_Recongnition import nucleus_recongnition
from haha.Align_Section import align_section
from haha.Assign_RNA_To_Cell import assign_rna_to_cell
from haha.Draw_Cell_Seg import draw_cell_seg
from haha.Cell_Type_Map import cell_type_map
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
    filter_mode = all_parameter["filter_mode"]
    top_value = all_parameter["top_value"]
    bottom_value = all_parameter["bottom_value"]
    gray_value_threshold = all_parameter["gray_value_threshold"]

    # 1.Detecting and filtering nucleus
    nucleus_recongnition(data_path, output_path, device, filter_mode, top_value, bottom_value)

    # 2.Align all section
    flag = all_parameter["section_align_flag"]
    if flag == 1:
        align_section(data_path, output_path, gray_value_threshold)

    # 3.Assign RNA to cell
    maximum_cell_radius = all_parameter['maximum_cell_radius']
    cell_express_min_number = all_parameter["cell_express_min_number"]
    assign_rna_to_cell(data_path, output_path, maximum_cell_radius, flag, cell_express_min_number)

    # 4.Draw cell segmentation result
    draw_cell_seg(data_path, output_path, flag)

    # 5.Cell type map
    cell_type_annotation_mode = all_parameter["cell_type_annotation_mode"]
    sc_data_path = all_parameter["sc_data_path"]
    if cell_type_annotation_mode == 1:
        cell_type_map(data_path, output_path, sc_data_path, device)
    else:
        all_section = os.listdir(data_path)
        for item in all_section:
            os.makedirs(os.path.join(output_path, item, "5_cell_type_result"), exist_ok=True)

    #
    all_section_cell_center = pd.DataFrame()
    all_section_cell_type = pd.DataFrame()
    all_result_save_path = os.path.join(output_path, "all_section_result")
    for item in os.listdir(data_path):
        cell_center_path = os.path.join(output_path, item, "3_gem/filtered_cell_center_coordinate.csv")
        cell_type_path = os.path.join(output_path, item, "5_cell_type_result/cell_type.csv")

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
