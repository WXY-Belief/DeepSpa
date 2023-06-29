import copy
import os
import time
import pandas as pd
import cv2
import numpy as np
import math
import shutil


def find_dapi_img(path):
    for file_name in os.listdir(path):
        if os.path.join(path, file_name).lower().endswith(('.png', '.jpg', '.tif')):
            return os.path.join(path, file_name)


def creat_dir(name, save_dir):
    for item_1 in name:
        os.makedirs(os.path.join(save_dir, str(item_1), "2_Aligned_result"), exist_ok=True)


def draw_point(point_path, img_path, save_path):
    coor = pd.read_csv(point_path, sep=",", header=0)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    coor_img = np.zeros(img.shape)
    for ig, item_5 in coor.iterrows():
        cv2.circle(coor_img, (int(item_5["col"]), int(item_5["row"])), radius=10, color=(255, 255, 255), thickness=-1)

    cv2.imwrite(save_path, coor_img)


def adjust_img_nucleus_rna(section_name, data_path, save_path):
    max_row_col = [0, 0]
    for item_2 in section_name:
        o_path = find_dapi_img(os.path.join(data_path, str(item_2)))
        o_dapi = cv2.imread(o_path, cv2.IMREAD_GRAYSCALE)
        if o_dapi.shape[0] > max_row_col[0]:
            max_row_col[0] = o_dapi.shape[0]
        if o_dapi.shape[1] > max_row_col[1]:
            max_row_col[1] = o_dapi.shape[1]

    row_add = int(max_row_col[0] / 2)
    col_add = int(max_row_col[1] / 2)
    max_row_col[0] += row_add
    max_row_col[1] += col_add

    for item_3 in section_name:
        base_dir = os.path.join(save_path, str(item_3), "2_Aligned_result")
        new_img = np.zeros(tuple(max_row_col))
        o_path = find_dapi_img(os.path.join(data_path, str(item_3)))
        o_dapi = cv2.imread(o_path, cv2.IMREAD_GRAYSCALE)
        row_shift = int(row_add / 2)
        col_shift = int(col_add / 2)
        new_img[row_shift:row_shift + o_dapi.shape[0], col_shift:col_shift + o_dapi.shape[1]] = o_dapi
        cv2.imwrite(os.path.join(base_dir, "adjust_img.PNG"), new_img)

        cell_center = pd.read_csv(
            os.path.join(save_path, str(item_3), "1_nucleus_recongnition_result/filtered_cell_center.csv"), sep=",",
            header=0)
        cell_center["row"] += row_shift
        cell_center["col"] += col_shift
        cell_center.to_csv(os.path.join(base_dir, "adjust_cell_center.csv"), sep=",", header=True, index=False)
        draw_point(os.path.join(base_dir, "adjust_cell_center.csv"), os.path.join(base_dir, "adjust_img.PNG"),
                   os.path.join(base_dir, "adjust_cell_center.PNG"))

        rna_coor = pd.read_csv(
            os.path.join(data_path, str(item_3), "rna_coordinate.csv"), sep=",",
            header=0)
        rna_coor["row"] += row_shift
        rna_coor["col"] += col_shift
        rna_coor.to_csv(os.path.join(base_dir, "adjust_rna_coordinate.csv"), sep=",",
                        header=True, index=False)
        draw_point(os.path.join(base_dir, "adjust_rna_coordinate.csv"), os.path.join(base_dir, "adjust_img.PNG"),
                   os.path.join(base_dir, "adjust_rna_coordinate.PNG"))


def find_centroid(section_name, save_path):
    in_centroid = [0, 0]
    in_all_centroid = []
    for item_4 in section_name:
        in_adjust_coor = pd.read_csv(os.path.join(save_path, str(item_4), "2_Aligned_result/adjust_cell_center.csv"),
                                     sep=",", header=0)
        row = int(sum(in_adjust_coor["row"]) / in_adjust_coor.shape[0])
        col = int(sum(in_adjust_coor["col"]) / in_adjust_coor.shape[0])
        in_all_centroid.append([row, col])
        if row > in_centroid[0] and col > in_centroid[1]:
            in_centroid[0] = row
            in_centroid[1] = col
        img = cv2.imread(os.path.join(save_path, str(item_4), "2_Aligned_result/adjust_img.PNG"))
        cv2.circle(img=img, center=(col, row), radius=20, color=(255, 255, 255),
                   thickness=-1, lineType=None, shift=None)
        cv2.imwrite(os.path.join(save_path, str(item_4), "2_Aligned_result/adjust_img_centroid.PNG"), img)
    return in_centroid, in_all_centroid


def draw_overlap_img(img_1_path, img_2_path, save_path):
    img_1 = cv2.imread(img_1_path, cv2.IMREAD_GRAYSCALE)
    img_2 = cv2.imread(img_2_path, cv2.IMREAD_GRAYSCALE)
    merge_img = np.zeros((img_1.shape[0], img_1.shape[1], 3))
    merge_img[:, :, 1] = img_1
    merge_img[:, :, 2] = img_2

    cv2.imwrite(save_path, merge_img)


def shift_img(section_name, in_centroid, in_all_centroid, save_path):
    for item_5 in section_name:
        base_dir = os.path.join(save_path, str(item_5), "2_Aligned_result")
        o_img = cv2.imread(os.path.join(base_dir, "adjust_img.PNG"), cv2.IMREAD_GRAYSCALE)
        row_shift = in_centroid[0] - in_all_centroid[section_name.index(item_5)][0]
        col_shift = in_centroid[1] - in_all_centroid[section_name.index(item_5)][1]
        m = np.float32([[1, 0, col_shift], [0, 1, row_shift]])
        in_shift_img = cv2.warpAffine(o_img, m, (o_img.shape[1], o_img.shape[0]))
        cv2.imwrite(os.path.join(base_dir, "shift_img.PNG"), in_shift_img)

        temp_img = copy.copy(o_img)
        cv2.circle(img=temp_img, center=(in_centroid[1], in_centroid[0]), radius=20, color=(255, 255, 255),
                   thickness=-1, lineType=None, shift=None)
        temp_img = cv2.warpAffine(temp_img, m, (temp_img.shape[1], temp_img.shape[0]))
        cv2.imwrite(os.path.join(base_dir, "shift_img_centroid.PNG"), temp_img)

        cell_center = pd.read_csv(os.path.join(base_dir, "adjust_cell_center.csv"), sep=",", header=0)
        cell_center["row"] += row_shift
        cell_center["col"] += col_shift
        cell_center.to_csv(os.path.join(base_dir, "shift_cell_center.csv"), sep=",", header=True, index=False)
        draw_point(os.path.join(base_dir, "shift_cell_center.csv"), os.path.join(base_dir, "shift_img.PNG"),
                   os.path.join(base_dir, "shift_cell_center.PNG"))

        rna_coor = pd.read_csv(os.path.join(base_dir, "adjust_rna_coordinate.csv"), sep=",", header=0)
        rna_coor["row"] += row_shift
        rna_coor["col"] += col_shift
        rna_coor.to_csv(os.path.join(base_dir, "shift_rna_coordinate.csv"), sep=",", header=True, index=False)
        draw_point(os.path.join(base_dir, "shift_rna_coordinate.csv"), os.path.join(base_dir, "shift_img.PNG"),
                   os.path.join(base_dir, "shift_rna_coordinate.PNG"))

        if int(item_5) == 1:
            continue
        else:
            path = os.path.join(save_path, str(int(item_5) - 1), "2_Aligned_result/shift_img_centroid.PNG")
            draw_overlap_img(path, os.path.join(base_dir, "shift_img.PNG"),
                             os.path.join(base_dir,
                                          str(int(item_5) - 1) + "_" + str(int(item_5)) + "_shift_img_centroid.PNG"))


def bina_img(img, threshold):
    img[img < threshold] = 0
    img[img >= threshold] = 255
    return img


def img_dilate(img, time):
    kernel = np.ones((9, 9), np.uint8)
    img = cv2.dilate(img, kernel, iterations=time)
    return img


def fiter_nosie(img, threshold):
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
    for stat in stats:
        if stat[4] < threshold:
            img[stat[1]:stat[1] + stat[3], stat[0]:stat[0] + stat[2]] = 0
    return img


def calculate_difference(img_1, img_2):
    in_distance = abs(img_2 - img_1)
    in_distance = int(in_distance.sum() / 255)
    return in_distance


def shift_nucleus_and_signal(nucleus_path, signal_path, add_row, add_col, save_dir):
    nucleus = pd.read_csv(nucleus_path, sep=",", header=0)
    nucleus["row"] += add_row
    nucleus["col"] += add_col
    nucleus.to_csv(os.path.join(save_dir, "shift_cell_center.csv"), sep=",", header=True, index=False)

    signal = pd.read_csv(signal_path, sep=",", header=0)
    signal["row"] += add_row
    signal["col"] += add_col
    signal.to_csv(os.path.join(save_dir, "shift_rna_coordinate.csv"), sep=",", header=True, index=False)


def rotate_nucleus_and_signal(coor_path, rotate_angle, rotate_center, save_path):
    coor = pd.read_csv(coor_path, sep=",", header=0)
    coor["row"] = rotate_center[0] - coor["row"]
    coor["col"] = coor["col"] - rotate_center[1]

    coor["z"] = np.sqrt(coor["row"] ** 2 + coor["col"] ** 2)

    coor["angle"] = np.arctan2(coor["row"], coor["col"]) + rotate_angle / 180 * math.pi

    coor["row"] = np.trunc(coor["z"] * np.sin(coor["angle"]))
    coor["col"] = np.trunc(coor["z"] * np.cos(coor["angle"]))

    coor["row"] = rotate_center[0] - coor["row"]
    coor["col"] = rotate_center[1] + coor["col"]

    coor.drop(["z", "angle"], axis=1, inplace=True)
    coor.to_csv(save_path, sep=",", header=True, index=False)


def align_section(data_path, output_path, gray_value_threshold):
    star_time = time.time()
    all_section = sorted(os.listdir(data_path))

    creat_dir(all_section, output_path)

    adjust_img_nucleus_rna(all_section, data_path, output_path)

    # find centroid
    centroid, all_centroid = find_centroid(all_section, output_path)

    # adjust centriod
    shift_img(all_section, centroid, all_centroid, output_path)

    iter_time_1 = 5
    iter_time_2 = 5
    area_threshold = 10000
    scale = 1 / 10
    for item_9 in all_section:
        output_base_dir = os.path.join(output_path, str(item_9), "2_Aligned_result")
        cell_center_path = os.path.join(output_base_dir, "shift_cell_center.csv")
        cell_center_save_path = os.path.join(output_base_dir, "rotate_cell_center.csv")
        rna_coordinate_path = os.path.join(output_base_dir, "shift_rna_coordinate.csv")
        rna_coordinate_save_path = os.path.join(output_base_dir, "rotate_rna_coordinate.csv")

        if item_9 == "1":
            align_img = cv2.imread(os.path.join(output_base_dir, "shift_img.PNG"))
            cv2.imwrite(os.path.join(output_base_dir, "align_img.PNG"), align_img)
            shutil.copy(cell_center_path, cell_center_save_path)
            shutil.copy(rna_coordinate_path, rna_coordinate_save_path)
            continue
        else:
            ref_img_path = os.path.join(output_path, str(int(item_9) - 1), "2_Aligned_result/align_img.PNG")
            need_alig_img_path = os.path.join(output_base_dir, "shift_img.PNG")
            ref_img = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)
            need_align_img = cv2.imread(need_alig_img_path, cv2.IMREAD_GRAYSCALE)

        # Binarization
        ref_img = bina_img(ref_img, gray_value_threshold)
        need_align_img = bina_img(need_align_img, gray_value_threshold)

        # dilation and erosion
        dilate_ref_img = img_dilate(ref_img, iter_time_1)
        filter_ref_img = fiter_nosie(dilate_ref_img, area_threshold)

        dilate_need_align_img = img_dilate(need_align_img, iter_time_1)
        filter_need_align_img = fiter_nosie(dilate_need_align_img, area_threshold)

        final_ref_img = img_dilate(filter_ref_img, iter_time_2)
        final_need_align_img = img_dilate(filter_need_align_img, iter_time_2)

        # resize img
        resize_ref_img = cv2.resize(final_ref_img, (int(final_ref_img.shape[1] * scale),
                                                    int(final_ref_img.shape[0] * scale)),
                                    interpolation=cv2.INTER_NEAREST)

        resize_need_align_img = cv2.resize(final_need_align_img, (int(final_need_align_img.shape[1] * scale),
                                                                  int(final_need_align_img.shape[0] * scale)),
                                           interpolation=cv2.INTER_NEAREST)

        # calculate diffence
        min_difference = calculate_difference(resize_ref_img, resize_need_align_img)

        totate_angle = 0
        for i in range(1, 361):
            m_rotate = cv2.getRotationMatrix2D((int(centroid[1] / 10), int(centroid[0] / 10)), i, 1)
            rotate_img = cv2.warpAffine(resize_need_align_img, m_rotate, (resize_need_align_img.shape[1],
                                                                          resize_need_align_img.shape[0]),
                                        flags=cv2.INTER_NEAREST, borderValue=0)

            difference = calculate_difference(resize_ref_img, rotate_img)
            if difference < min_difference:
                min_difference = difference
                totate_angle = i

        o_img_1 = cv2.imread(need_alig_img_path, cv2.IMREAD_GRAYSCALE)
        rotate_o = cv2.getRotationMatrix2D((centroid[1], centroid[0]), totate_angle, 1)
        rotate_img_o = cv2.warpAffine(o_img_1, rotate_o, (o_img_1.shape[1], o_img_1.shape[0]),
                                      flags=cv2.INTER_NEAREST, borderValue=0)
        cv2.imwrite(os.path.join(output_base_dir, "align_img.PNG"), rotate_img_o)

        rotate_nucleus_and_signal(cell_center_path, totate_angle, centroid, cell_center_save_path)
        rotate_nucleus_and_signal(rna_coordinate_path, totate_angle, centroid, rna_coordinate_save_path)

        draw_nucleus_save_path = os.path.join(output_base_dir, "rotate_cell_center.PNG")
        draw_signal_save_path = os.path.join(output_base_dir, "rotate_rna_coordinate.PNG")
        draw_point(cell_center_save_path, ref_img_path, draw_nucleus_save_path)
        draw_point(rna_coordinate_save_path, ref_img_path, draw_signal_save_path)

        ref_img_1 = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)
        merge_img = np.zeros((o_img_1.shape[0], o_img_1.shape[1], 3))
        merge_img[:, :, 1] = ref_img_1
        merge_img[:, :, 2] = rotate_img_o

        cv2.imwrite(os.path.join(output_base_dir, "debug_merge.PNG"), merge_img)

    print("alinged section finished, runtime:", time.time() - star_time)
