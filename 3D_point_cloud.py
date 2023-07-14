from haha.Draw_Cell_Point_Cloud import draw_3d
import toml
import argparse

if __name__ == "__main__":
    # input parameter
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=str, default="Configration.toml")
    parser.add_argument("--m", type=str, default="all")
    args = parser.parse_args()

    # Load configuration file
    all_parameter = toml.load(args.c)
    output_path = all_parameter["output_path"]  # Directory where the output results are saved

    draw_3d(output_path, args.m)
