## Environment
python=3.7  
torch=1.9.1+cu111  
mmcv-full=1.6.1  
## Data prepare
We downloaded the dataset from the official website of dsb2018 and tissuenet, and crop the training set of dsb2018 to 256x256. We put the processed data on https://figshare.com/account/articles/24117366. Users can download and store it in the ./data folder, and unzip it. 


- data
  - dsb2018
    - images
    - labels
  - tissuenet_1.0
    - images
    - labels
  - CoNSep
    - images
    - labels
  - PanNuke
    - images
    - labels


## Train

`./tools/dist_train.sh configs/dsb.py 1`

`./tools/dist_train.sh configs/tissuenet.py 1`

`./tools/dist_train.sh configs/consep.py 1`

`./tools/dist_train.sh configs/pannuke.py 1`

### Eval

`python ./tools/test.py configs/dsb.py <model_path> --eval all`



## Refernence

[open-mmlab/mmsegmentation: OpenMMLab Semantic Segmentation Toolbox and Benchmark. (github.com)](https://github.com/open-mmlab/mmsegmentation)

[lhoyer/DAFormer: [CVPR22\] Official Implementation of DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation (github.com)](https://github.com/lhoyer/DAFormer)

[MouseLand/cellpose: a generalist algorithm for cellular segmentation with human-in-the-loop capabilities (github.com)](https://github.com/MouseLand/cellpose)

