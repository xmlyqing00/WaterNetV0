# WaterNetV0

This is a package for water level estimation, named **WaterNetV0**. For using these scripts, open a Terminal and type the filename to run.

## 1 Environment
All python scripts are written and tested under Ubuntu 18.04 and Python 3.6

The required libraries and scripts could be simply installed by
```bash
pip install -r requirements.txt
```
Details of the version of the required libraries could be found in the file `requirements.txt`

## 2 Dataset and Pretrained Model

The dataset uploaded to Kaggle as `water_v1`. Click [here](https://www.kaggle.com/gvclsu/water-segmentation-dataset?select=water_v1) to download the dataset.
We also provide the pretrained model `checkpoint_99.pth.tar`. Click [here]() to download.


## 3 Usage

### 3.1 Evaluation

Segment water area from the test video frames. The results will be saved into `data/` by default. 
```bash
python3 test_model.py -c /path/to/checkpoint.pth.tar -i /path/to/water_v1/
```
More options: `-o`, Path to the output segmentations (default: data/raw/). `--name`, Test video name (default: houston).

Apply temporal constraint and prior constraint, then estimate the water level. The results will be saved into the `data/` folder by default.
```bash
python3 estimate_waterlevel.py
```
More options: `--out-dir`, Path to the output dir (default: data/). `--anchor-x`, Referece point X. `--anchor-y`, Referece point Y.　`--ori-h`, Original height.
    

### 3.2 Visualization
Add the estimated water masks to the original frames.
```bash
python3 plot_overlay
```
More options: `--img-dir`, Path to the input image folder. `--seg-dir`, Path to the segmentation folder. `--out-dir`, Path to the output overlay folder.

Compare the estiamted water level with the groundtruth, and compare the results w/ or w/o prior constraint.
```bash
python3 plot_waterlevel.py
```
Note that we attach the groundtruth data of the houston flood in `data/buffalo_gt.csv`.  For the all above scripts, you can type `--help` to ses the parameters that can be used.

### 3.3 Retrain the model
Retrain the model,
```bash
python3 train_model.py
```
optional arguments:
```bash
  -h, --help            show this help message and exit
  --start-epoch N       Manual epoch number (useful on restarts, default 0).
  --total-epochs N      Number of total epochs to run (default 100).
  --lr LR, --learning-rate LR
                        Initial learning rate.
  --resume PATH         Path to latest checkpoint (default: none).
  --dataset PATH        Path to the training dataset
  --modelpath PATH      Path to the models.
```

## References:

> [1] Russell B C, Torralba A, Murphy K P, et al. LabelMe: a database and web-based tool for image annotation[J]. International journal of computer vision, 2008, 77(1-3): 157-173.

> [2] Long J, Shelhamer E, Darrell T. Fully convolutional networks for semantic segmentation[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2015: 3431-3440.