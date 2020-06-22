# WaterLevelEstimation

This is a package for water level estimation, named **LSUWaterLevelEstimation**. It contains two folders `FCN` and `utils`. For using these scripts, open a Terminal and type the filename to run.

## Environment
All python scripts are written and tested under Ubuntu 18.04 and Python 3.6

The required libraries and scripts could be simply installed by 
```
    pip install -r requirements.txt
```
Details of the version of the required libraries could be found in the file `requirements.txt`

## Dataset and Pretrained Model

The dataset is split into training data (`water_v1.zip`), and evaluation dataset (`test_videos.zip`). We also provide the pretrained model (`checkpoint_58.pth.tar`).
Click [here](https://www.dropbox.com/sh/yk39hpqwnzauv02/AAA_IYacZf_bEbcURj-PQXIra?dl=0) to download the dataset and the pretrained model.

You should put the pretrained model at the path `LSUWaterSegmentation/data/models/checkpoint_58.pth.tar`.

You should put the contents of `test_videoes.zip` at `LSUWaterSegmentation/data/models/houston_small`.

## Usage

### Evaluate the model

To evaluate the model, open a Terminal and type
```
    cd FCN
    python3 test_model.py
```
Type `--help` to ses the parameters that can be used.
### Retrain the model
To retrain the model, open a Terminal and type
```
    cd FCN
    python3 train_model.py
```
Type `--help` to ses the parameters that can be used.

## Documentation

### Scripts in FCN
This folder contains the scripts of network architecture, training and evaluation.

- `cmp_estimations.py`: Compare and show the results of two water levels.
- `drop_error_estimation.py`: Compute the water level and drop errors.
- `estimate_waterlevel.py`: Estimate the water level from the water segmentations of the frames.
- `model.py`: Network architecture.
- `post_processing`: Temporal and domain knowledge constraints.
- `test_model.py`: Scripts for evaluation.
- `train_model.py`: Scripts for training the model.

### Scripts in Utils

This folder contains scripts for preparing dataset, preprocessing and postprocessing.

- `add_prefix.py`: Add prefix to the path
- `AvgMeter.py`: Log the model performance when training the model.
- `cvt_images_to_overlays.py`: Add the segmentation masks on the original frames.
- `cvt_images_to_video,py`: Compose the frames to a video.
- `cvt_labelme_to_collection.py`: User annotations are created by LabelMe. This script help format the results of LabelMe.
- `cvt_object_label.py`: Change the color representations of the annotations.
- `dataset.py`: Class for dataset I/O.
- `format_file_names.py`: Scripts for formatting the filenames.
- `laplacian_smooth.py`: Script for laplacian smooth.


## References:

[1] Russell B C, Torralba A, Murphy K P, et al. LabelMe: a database and web-based tool for image annotation[J]. International journal of computer vision, 2008, 77(1-3): 157-173.

[2] Long J, Shelhamer E, Darrell T. Fully convolutional networks for semantic segmentation[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2015: 3431-3440.