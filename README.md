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
### format of input data
The input data include two part including DAPI images and RNA molecules infomation that include gene name and coordinate.  
- DAPI images  
- RNA molecules infomation   
<div align="center">
  
| index | gene | row | col | num |
| ------- | ------- | ------- | ------- | ------- |
| 0 | Cdhr1 | 1 | 100 | 2 |
| 1 | Gad1 | 4 | 239 | 3 |
| 2 | Cdhr1 | 100 | 590 | 5 |

</div>

-`gene`: gene name  
-`row`: row coordinate of transcript  
-`col`: column coordinate of transcript  
-`num`: number of transcript in spot for based-NGS spatial transcriptomics. The based-imaging spatial transcriptomics doesn't need "num".  
