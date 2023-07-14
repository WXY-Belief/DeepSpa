# DeepSpa
## 1.Introduction
This repository include an available learning-based cell segmentation tool for DeepSpa that analyse subcellular resolution spatial transcriptomics.
## 2.Prerequisites
To install requirements:  
```
pip install -r requirements.txt
```  
- Python 3.8.13  
- GPU Memory: 3GB+  
## 3.Tutorial
### Directory structure of input data
The sections are numerically named, starting from 1 for the top layer. The DAPI images are in TIFF, PNG, or JPEG format, while the RNA coordinate file must be in CSV format.The directory structure is as follows:  


- all
  - 1 
    - DAPI.tif
    - rna_coordinate.csv
  - 2 
    - DAPI.tif 
    - rna_coordinate.csv
  - 3
    - DAPI.tif 
    - rna_coordinate.csv  


### Format of input data
(1) DAPI.tif: DAPI images  

(2) rna_coordinate.csv: RNA molecules infomation   
<div align="center">
  
| index | gene | row | col | num |
| ------- | ------- | ------- | ------- | ------- |
| 0 | gene 1 | 1 | 100 | 2 |
| 1 | gene 2 | 4 | 239 | 3 |
| 2 | gene 1 | 100 | 590 | 5 |

</div>


- `gene`: gene name
  
- `row`: row coordinate of transcript
  
- `col`: column coordinate of transcript
  
- `num`: number of transcript in spot for based-NGS spatial transcriptomics. The based-imaging spatial transcriptomics doesn't need "num".
  
(3) sc_data_path: The Sc-RNA data ,csv file , used in tangram for cell type annotation.
<div align="center">
  
| index | cell_type | gene 1 | gene 2 | ...  | gene n |
| ------- | ------- | ------- | ------- | ------- | ------- |
| 0 | cell_type 1 | 0 | 10 | ... | 2 |
| 1 | cell_type 2 | 4 | 23 | ... | 4 |
| 2 | cell_type 1 | 10 | 50 | ... | 8 |

</div>  


### Input parameter  

The Configration.toml include all parameter in runing. 

- `data_path`: the directory of input data

- `output_path`: the directory of output data
  
- `seg_threshold`: segmnetation threshold. lower values will generate more cell.

- `seg_mode`: segmentation mode. 1: original DAPI as input. 2:The DAPI image is divided into small images of 512 * 512 as input

- `device`: Using GPU or CPU, GPU is faster. 

- `section_align_flag`: whether need to align section. 0:NO 1:YES

- `filter_mode`: Nuclear filtration mode. 1: Filter by diameter 2: Filter by cell distribution.

- `top_value`: 1: Filter by diameter. The maximum radius of a nucleus, in px. or  2: Filter by cell distribution. Top percentage.ascend order

- `bottom_value`: 1: Filter by diameter. The minimum radius of a nucleus, in px. or 2: Filter by cell distribution. Bottom percentage.ascend order

- `gray_value_threshold`: # The DAPI image will be subtracted from this value to reduce impact of overexposure.

- `maximum_cell_radius`: # The maximum radius of a cell, in px, control size of cell.

- `cell_express_min_number`: # The lowest value of cell expression, and cells below this value will be filtered.

- `cell_type_annotation_mode`: # cell type nnotation mode. 1:tangram 2:clustering

- `sc_data_path`: # The single cell expression matrix applied in tangram.

- `K`: # Number of neighboring cells in find anatomic region

- `anatomic_region_num`: # The number of anatomic region

### Running
After selecting all parameter value, the cell segmentation will be executed by running the following command.

```
python main_1.py --c ./Configration.toml
```

If `cell_type_annotation_mode=2`,using traditional clustering for cell type annotation ,you will need to execute the following command.

```
python main_2.py --c ./Configration.toml
```

You can use point cloud by running the following command in local terminal for 3D reconstruction including 3D surface of tissue/organ and 3D visualization of cell type and anatomic region.
```
python 3D_point_cloud.py --c ./Configration.toml --m all
```
Notice: The code runs on the server by default. If running on a local terminal, please pay attention to the path format.
