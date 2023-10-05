import os
import cv2
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from haha.valis import registration, valtils
from skimage import transform as tf
import pyvips

Image.MAX_IMAGE_PIXELS = None


def draw_point(point, img, save_path):
    new_img = np.zeros((img.shape[0], img.shape[1], 3))
    new_img[:, :, 0] = img
    point_img = np.zeros((img.shape[0], img.shape[1]))
    for idx, item in point.iterrows():
        cv2.circle(img=point_img, center=(int(item["col"]), int(item["row"])), radius=5, color=(255, 0, 0),
                   thickness=-1)
    new_img[:, :, 1] = point_img
    cv2.imwrite(os.path.join(save_path, "rna.PNG"), new_img)


def preprocessed_images(path, save_path, Grayscale_threshold, padding):
    all_section = sorted([int(item) for item in os.listdir(path)])

    # step 1： revise the size of images
    max_img_shape = [0, 0]
    for item in all_section:
        dapi_path = os.path.join(path, str(item), "DAPI.PNG")
        dapi = Image.open(dapi_path).convert("L")

        if dapi.height > max_img_shape[0]:
            max_img_shape[0] = dapi.height
        if dapi.width > max_img_shape[1]:
            max_img_shape[1] = dapi.width

    max_img_shape[0] += padding
    max_img_shape[1] += padding
    step_1_save_path = os.path.join(save_path, "revised_images")
    os.makedirs(step_1_save_path, exist_ok=True)

    for item in all_section:
        # dapi = np.array(Image.open(os.path.join(path, str(item), "DAPI.PNG")))
        dapi = cv2.imread(os.path.join(path, str(item), "DAPI.PNG"), cv2.IMREAD_GRAYSCALE)
        new_dapi = np.zeros(max_img_shape)
        new_dapi[padding:dapi.shape[0] + padding, padding:dapi.shape[1] + padding] = dapi
        new_dapi[new_dapi < Grayscale_threshold] = 0
        print("new_dapi:", new_dapi.shape, np.max(new_dapi))

        cv2.imwrite(os.path.join(step_1_save_path, str(item) + ".PNG"), new_dapi)

    # step 2：transform the formate of images
    step_2_save_path = os.path.join(save_path, "transformed_images")
    os.makedirs(step_2_save_path, exist_ok=True)
    for item in all_section:
        ori_dapi_path = os.path.join(step_1_save_path, str(item) + ".PNG")
        transformed_path = os.path.join(step_2_save_path, str(item) + ".tiff")
        command = "vips tiffsave " + ori_dapi_path + " " + transformed_path + " --tile --pyramid"
        print(command)
        os.system(command)
    return all_section


def generate_alinged_need_dir(results_dst_dir, aligned_need_dir, aligned_path, section):
    # construct the folder aligned
    shutil.copyfile(os.path.join(results_dst_dir, "transformed_images", str(section) + ".tiff"),
                    os.path.join(aligned_need_dir, str(2) + ".tiff"))
    if section == 2:
        shutil.copyfile(os.path.join(results_dst_dir, "transformed_images", str(section - 1) + ".tiff"),
                        os.path.join(aligned_need_dir, str(1) + ".tiff"))
    else:
        last_aligned_path = os.path.join(aligned_need_dir, str(1) + ".tiff")
        command = "vips tiffsave " + os.path.join(aligned_path,
                                                  "DAPI.PNG") + " " + last_aligned_path + " --tile --pyramid"
        os.system(command)


def transform_rna(save_path, M, file_path, padding, shift, crop):
    transform = np.array([[M[0], M[1]],
                          [M[2], M[3]]])

    # rna_coor = pd.read_csv(rna_file, sep=",", header=0)
    # np_rna_coor = rna_coor[["row", "col"]].to_numpy()
    #
    # # adding padding
    # np_rna_coor += np.array([padding - int(shift[1]), padding - int(shift[0])])
    #
    # # scale and rotation
    # aligned_np_rna_coor = np.dot(np_rna_coor, transform)
    # aligned_np_rna_coor -= np.array([int(crop[1]), int(crop[0])])
    #
    # aligned_rna_coor = pd.DataFrame(aligned_np_rna_coor)
    # aligned_rna_coor.rename({0: "row", 1: "col"}, axis=1, inplace=True)
    # aligned_rna_coor["gene"] = rna_coor["gene"]
    # aligned_rna_coor.to_csv(os.path.join(save_path, "aligned_rna_coordinate.csv"), sep=",", header=True, index=False)
    #
    # draw_point(aligned_rna_coor, bk_img, save_path)
    #
    # # transform nucleus coor
    # nucleus_coor = pd.read_csv(nucleus_coor_file, sep=",", header=0)
    # np_nucleus_coor = nucleus_coor[["row", "col"]].to_numpy()
    #
    # np_nucleus_coor += np.array([padding - int(shift[1]), padding - int(shift[0])])
    # aligned_np_nucleus_coor = np.dot(np_nucleus_coor, transform)
    # aligned_np_nucleus_coor -= np.array([int(crop[1]), int(crop[0])])
    #
    # aligned_nucleus_coor = pd.DataFrame(aligned_np_nucleus_coor)
    # aligned_nucleus_coor.rename({0: "row", 1: "col"}, axis=1, inplace=True)
    # aligned_nucleus_coor["area"] = nucleus_coor["area"]
    # aligned_nucleus_coor["cell_index"] = nucleus_coor["cell_index"]
    # aligned_nucleus_coor.to_csv(os.path.join(save_path, "aligned_cell_center.csv"), sep=",", header=True, index=False)

    #
    coor_file = pd.read_csv(file_path, sep=",", header=0)
    coor = coor_file[["row", "col"]].to_numpy()
    coor += np.array([padding - int(shift[1]), padding - int(shift[0])])
    aligned_coor = np.dot(coor, transform)
    aligned_coor -= np.array([int(crop[1]), int(crop[0])])

    coor_file.drop(["row", "col"], axis=1, inplace=True)
    coor_file["row"] = aligned_coor[:, 0]
    coor_file["col"] = aligned_coor[:, 1]

    coor_file.to_csv(save_path, sep=",", header=True, index=False)


def sb_step(registrar, save_path):
    # def cnames_from_filename(src_f):
    #     f = valtils.get_name(src_f)
    #     return ["DAPI"] + f.split(" ")
    #
    # channel_name_dict = {f: cnames_from_filename(f) for
    #                      f in registrar.original_img_list}

    dst_f = os.path.join(save_path, registrar.name, registrar.name + ".tiff")
    merged_img, channel_names, ome_xml = registrar.warp_and_merge_slides(dst_f,
                                                                         channel_name_dict=None,
                                                                         drop_duplicates=True,
                                                                         non_rigid=False, crop=True)
    return merged_img


def align_section(data_path, output_path, grayscale_threshold, padding: int = 1000):
    result_save_path = os.path.join(output_path, "aligned_temp")

    # Convert the image data format using VIPS.
    all_section = preprocessed_images(data_path, result_save_path, grayscale_threshold, padding)

    # calculate the transform matrix
    print("all section:", all_section)

    registration_result_path = os.path.join(result_save_path, "registration_temp_file")
    os.makedirs(registration_result_path, exist_ok=True)

    aligned_debug_path = os.path.join(result_save_path, "aligned_debug_img")
    os.makedirs(aligned_debug_path, exist_ok=True)

    final_aligned_path = os.path.join(result_save_path, "aligned_result")
    os.makedirs(final_aligned_path, exist_ok=True)

    print("--------strating aligned--------")

    for item in all_section:
        print("section:", item)
        this_section_aligned_save_path = os.path.join(output_path, str(item), "3_aligned_result")
        os.makedirs(this_section_aligned_save_path, exist_ok=True)

        if item == 1:
            src_path = os.path.join(output_path, str(item), "2_gem")
            shutil.copyfile(os.path.join(data_path, str(item), "rna_coordinate.csv"),
                            os.path.join(this_section_aligned_save_path, "aligned_rna_coordinate.csv"))
            shutil.copyfile(os.path.join(src_path, "filtered_cell_center_coordinate.csv"),
                            os.path.join(this_section_aligned_save_path, "aligned_cell_center_coordinate.csv"))
            shutil.copyfile(os.path.join(src_path, "filtered_RNA_and_nearest_cell.csv"),
                            os.path.join(this_section_aligned_save_path, "aligned_RNA_and_nearest_cell.csv"))
            shutil.copyfile(os.path.join(result_save_path, "revised_images", str(item)+".PNG"),
                            os.path.join(this_section_aligned_save_path, "DAPI.PNG"))
            continue
        this_section_debug_save_path = os.path.join(aligned_debug_path, str(item))
        os.makedirs(this_section_debug_save_path, exist_ok=True)

        aligned_need_dir = os.path.join(result_save_path, "aligned_need_dir", str(item))
        os.makedirs(aligned_need_dir, exist_ok=True)

        # construct the folder aligned
        generate_alinged_need_dir(result_save_path, aligned_need_dir,
                                  os.path.join(output_path, str(item - 1), "3_aligned_result"), item)

        registration_temp_path = os.path.join(registration_result_path, str(item))
        os.makedirs(registration_temp_path, exist_ok=True)
        registrar = registration.Valis(aligned_need_dir, registration_temp_path, reference_img_f="1.tiff",
                                       align_to_reference=True, imgs_ordered=True, non_rigid_registrar_cls=None,
                                       do_rigid=True)

        rigid_registrar, non_rigid_registrar, error_df = registrar.register()
        merged_img = sb_step(registrar, registration_result_path)

        image_array = np.ndarray(buffer=merged_img.write_to_memory(), dtype=np.uint8,
                                 shape=[merged_img.height, merged_img.width, merged_img.bands])

        # save the aligned img
        new_img = np.zeros((image_array.shape[0], image_array.shape[1], 3))
        new_img[:, :, 0] = image_array[:, :, 0]
        new_img[:, :, 1] = image_array[:, :, 1]
        cv2.imwrite("/".join([this_section_debug_save_path, "aligned_merge.PNG"]), new_img)
        cv2.imwrite("/".join([this_section_aligned_save_path, "DAPI.PNG"]), image_array[:, :, 1])

        # transform RNA and DAPI
        if item == 2:
            ref_img = cv2.imread("/".join([aligned_need_dir, "1.tiff"]), cv2.IMREAD_GRAYSCALE)
        else:
            ref_img = cv2.imread("/".join([output_path, str(item - 1), "3_aligned_result/DAPI.PNG"]),
                                 cv2.IMREAD_GRAYSCALE)

        ori_img_py = pyvips.Image.new_from_file(os.path.join(result_save_path, "revised_images", str(item) + ".PNG"))
        tx_ty = pd.read_csv("./tx_ty.csv", sep=",", header=0)
        tx = tx_ty["tx"].values
        ty = tx_ty["ty"].values
        M = list(pd.read_csv("./final_m.csv", sep=",", header=0)["M"].values)
        out_shape = registrar.aligned_slide_shape_rc
        slide_bbox_xywh = registrar.slide_dict["2"].slide_bbox_xywh

        rotate_info = pd.DataFrame()
        rotate_info["tx"] = tx
        rotate_info["ty"] = ty
        rotate_info["M"] = str(M)
        rotate_info["out_shape"] = str([out_shape[0], out_shape[1]])
        rotate_info["slide_bbox_xywh"] = str([slide_bbox_xywh[0], slide_bbox_xywh[1]])
        rotate_info["padding"] = padding
        rotate_info.to_csv("/".join([this_section_debug_save_path, "rotation_infomation.csv"]), sep=",", header=True,
                           index=False)

        aligned_img = ori_img_py.affine(M, oarea=[0, 0, out_shape[1], out_shape[0]],
                                        interpolate=pyvips.Interpolate.new("bicubic"),
                                        idx=-tx,
                                        idy=-ty,
                                        premultiplied=True,
                                        background=[0],
                                        extend=pyvips.enums.Extend.BLACK
                                        )
        warp_img = aligned_img.extract_area(*slide_bbox_xywh)
        warp_img.write_to_file("temp.png")
        aligned_img = cv2.imread("temp.png", cv2.IMREAD_GRAYSCALE)

        merge_py_img_2 = np.zeros((ref_img.shape[0], ref_img.shape[1], 3))
        merge_py_img_2[:, :, 0] = ref_img
        merge_py_img_2[:, :, 1] = aligned_img
        cv2.imwrite("/".join([this_section_debug_save_path, "pv_aligned_img.PNG"]), merge_py_img_2)

        # transformer rna and nucleus coor
        shift = (tx, ty)
        rna_file = os.path.join(data_path, str(item), "rna_coordinate.csv")
        save_path_1 = os.path.join(this_section_aligned_save_path, "aligned_rna_coordinate.csv")
        transform_rna(save_path_1, M, rna_file, padding, shift, slide_bbox_xywh[0:2])

        nucleus_coor = os.path.join(output_path, str(item), "2_gem/filtered_cell_center_coordinate.csv")
        save_path_2 = os.path.join(this_section_aligned_save_path, "aligned_cell_center_coordinate.csv")
        transform_rna(save_path_2, M, nucleus_coor, padding, shift, slide_bbox_xywh[0:2])

        rna_near_cell = os.path.join(output_path, str(item), "2_gem/filtered_RNA_and_nearest_cell.csv")
        save_path_3 = os.path.join(this_section_aligned_save_path, "aligned_RNA_and_nearest_cell.csv")
        transform_rna(save_path_3, M, rna_near_cell, padding, shift, slide_bbox_xywh[0:2])

    registration.kill_jvm()


if __name__ == "__main__":
    data_apth = "../ISS_DATA"
    result_save_path = "../ISS_registration_results"
    Grayscale_threshold_a = 5
    padding_a = 1000
    align_section(data_apth, result_save_path, Grayscale_threshold_a, padding_a)
