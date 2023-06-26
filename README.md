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
The directory structure entered is as follows:  

  ├── all  
  │  ├── 1  
  │       ├── rna_coordinate.csv  
  │       └── DAPI.tif  
  │   ├── 2  
  │       ├── rna_coordinate.csv  
  │       └── DAPI.tif  
  │   └── 3  
  │       ├── rna_coordinate.csv  
  │       └── DAPI.tif  


### Format of input data
The input data include two part including DAPI images and RNA molecules infomation that include gene name and coordinate.  
(1) DAPI images  
(2) RNA molecules infomation   
<div align="center">
  
| index | gene | row | col | num |
| ------- | ------- | ------- | ------- | ------- |
| 0 | Cdhr1 | 1 | 100 | 2 |
| 1 | Gad1 | 4 | 239 | 3 |
| 2 | Cdhr1 | 100 | 590 | 5 |

</div>

- `gene`: gene name
  
- `row`: row coordinate of transcript
  
- `col`: column coordinate of transcript
  
- `num`: number of transcript in spot for based-NGS spatial transcriptomics. The based-imaging spatial transcriptomics doesn't need "num".  

### Input parameter
The Configration.toml include all parameter in runing.
- `data_path`: the data path

- `output_path` = "" # the output path

- `device` = "GPU" # Devices used

- `section_align_flag` = 0 # whether need to align section. 0:NO 1:YES

- `filter_mode` = 1   # Nuclear filtration mode. 1: Filter by diameter 2: Filter by cell distribution.

- `top_value` = 10  # 1: Filter by diameter. The maximum radius of a nucleus, in px. or  2: Filter by cell distribution. Top percentage.ascend order

- `bottom_value` = 0  #  1: Filter by diameter. The minimum radius of a nucleus, in px. or 2: Filter by cell distribution. Bottom percentage.ascend order

- `gray_value_threshold` = 5  # The DAPI image will be subtracted from this value

- `maximum_cell_radius` = 50  # The maximum radius of a cell, in px

- `cell_express_min_number` = 5 # The lowest value of cell expression, and cells below this value will be filtered

- `cell_type_annotation_mode` = 1   # cell type nnotation mode. 1:tangram 2:clustering

- `sc_data_path` = ""   # The single cell expression matrix

- `K` = 30 # Number of neighboring cells in find anatomic region

- `anatomic_region_num` = 10 # The number of anatomic region

- `draw_3d_mode` = 1 # the way of 3d exhibition. 1: ploty 2: point cloudoutput_path = "" # the output path

