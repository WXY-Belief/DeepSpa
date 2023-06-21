import time

import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")

"""
本模块的作用：过滤掉过大或过小的细胞核，
"""


def nucleus_radius_filter(input_path, save_path, maximum_cell_radius, minimum_cell_radius):
    start_time = time.time()
    filter_coordinate_result = os.path.join(save_path, "filtered_cell_center_and_area.csv")
    coordinate_file = pd.read_csv(os.path.join(input_path, "cell_center_and_area.csv"), sep=",", header=0,
                                  index_col=None)

    # filtering nucleus
    coordinate_file1 = coordinate_file[
        (coordinate_file["area"] > (minimum_cell_radius ** 2) * 3.14) & (
                coordinate_file["area"] < (maximum_cell_radius ** 2) * 3.14)]

    # add cell index
    cell_index = list(range(1, coordinate_file1.shape[0] + 1))
    coordinate_file1["cell_index"] = cell_index

    # save to file
    coordinate_file1.to_csv(filter_coordinate_result, sep=",", header=True, index=False)

    print("filtering nucleus finished, runtime：", time.time() - start_time)


def distribution_filter(input_path, save_path, top_percentage_value, bottom_percentage_value):
    start_time = time.time()
    filter_coordinate_result = os.path.join(save_path, "filtered_cell_center_and_area.csv")
    coordinate_file = pd.read_csv(os.path.join(input_path, "cell_center_and_area.csv"), sep=",", header=0,
                                  index_col=None)

    row_num = coordinate_file.shape[0]
    top_value1 = int(row_num / 100 * top_percentage_value)
    bottom_value1 = int(row_num / 100 * bottom_percentage_value)

    # filtering nucleus
    coordinate_file1 = coordinate_file.sort_values("area").iloc[top_value1:]
    if bottom_value1 == 0:
        coordinate_file2 = coordinate_file1.iloc[:-1]
    else:
        coordinate_file2 = coordinate_file1.iloc[:-bottom_value1]

    # add cell index
    cell_index = list(range(1, coordinate_file2.shape[0] + 1))
    coordinate_file2["cell_index"] = cell_index

    coordinate_file2.to_csv(filter_coordinate_result, sep=",", header=True, index=False)

    print("filtering nucleus finished, runtime：", time.time() - start_time)
