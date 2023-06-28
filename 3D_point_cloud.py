from haha.Draw_Cell_Point_Cloud import draw_3d
import toml

if __name__ == "__main__":
    draw_type = "anatomic_region"
    # Load configuration file
    configration_file_path = "./Configration.toml"
    all_parameter = toml.load(configration_file_path)
    output_path = all_parameter["output_path"]  # Directory where the output results are saved
    # output_path = "C:/Users/24667/Desktop/3D_ISS_result"

    draw_3d(output_path, draw_type)
