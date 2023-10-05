import time
import toml
import argparse
from haha.Nuclues_Recongnition import nucleus_recongnition
from haha.Nucleus_Recongnition_Cut_img import nucleus_recongnition_cut_img
from haha.Align_Section import align_section
from haha.Assign_RNA_To_Cell import assign_rna_to_cell
from haha.Draw_Cell_Seg import draw_cell_seg
from main_2 import main

if __name__ == "__main__":
    star_time = time.time()

    # input parameter
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=str, default="Configration.toml")
    args = parser.parse_args()
    # Load configuration file
    all_parameter = toml.load(args.c)

    data_path = all_parameter["data_path"]  # Directory where the data is located
    output_path = all_parameter["output_path"]  # Directory where the output results are saved
    device = all_parameter["device"]
    filter_mode = all_parameter["filter_mode"]
    top_value = all_parameter["top_value"]
    bottom_value = all_parameter["bottom_value"]
    gray_value_threshold = all_parameter["gray_value_threshold"]
    seg_threshold = all_parameter["seg_threshold"]
    seg_mode = all_parameter["seg_mode"]

    # # 1.Detecting and filtering nucleus
    # if seg_mode == 1:
    #     nucleus_recongnition(data_path, output_path, device, filter_mode, top_value, bottom_value, seg_threshold)
    # else:
    #     nucleus_recongnition_cut_img(data_path, output_path, device, filter_mode, top_value, bottom_value,
    #                                  seg_threshold)
    #
    # # 2.Assign RNA to cell
    # maximum_cell_radius = all_parameter['maximum_cell_radius']
    # cell_express_min_number = all_parameter["cell_express_min_number"]
    # cell_express_max_number = all_parameter['cell_express_max_number']
    # experiment_mode = all_parameter["experiment_mode"]
    # assign_rna_to_cell(data_path, output_path, maximum_cell_radius, cell_express_min_number,
    #                    cell_express_max_number, experiment_mode, device)
    #
    # # 3.Align all section
    flag = all_parameter["section_align_flag"]
    if flag == 1:
        align_section(data_path, output_path, gray_value_threshold)

    # 4.Draw cell segmentation result
    draw_cell_seg(data_path, output_path, flag)

    # if all_parameter["cell_type_annotation_mode"] == 1:
    #     main(args.c)
