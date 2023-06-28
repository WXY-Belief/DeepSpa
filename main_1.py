import os
import time
import toml
import subprocess
from haha.Nuclues_Recongnition import nucleus_recongnition
from haha.Align_Section import align_section
from haha.Assign_RNA_To_Cell import assign_rna_to_cell
from haha.Draw_Cell_Seg import draw_cell_seg
from main_2 import main
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
    pretrain_weight = all_parameter["pretrain_weight"]

    # 1.Detecting and filtering nucleus
    nucleus_recongnition(data_path, output_path, device, filter_mode, top_value, bottom_value, pretrain_weight)

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

    if all_parameter["cell_type_annotation_mode"] == 1:
        main()



