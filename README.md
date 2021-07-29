# DIS-YOLO
In this repository, a fully convolutional model is proposed to simultaneously detect and group image pixels for multiple concrete surface defects, i.e. instance-level defect segmentation.

# Requirements
Python 3.x, TensorFlow > 1.0 and other common packages.

# Network architecture
The proposed network consists of a fast single-stage detector YOLOv3 for box-level defect detection, and an optimized mask subnet that predicts a set of position-sensitive score maps for assembling defect masks.

<div align=center><img src="https://github.com/ZHANGKEON/DIS-YOLO/blob/master/output/architecture.jpg"/></div>

# Concrete defect [dataset]( https://drive.google.com/file/d/1UbAnTFQWShtuHlGEvYYZ4TP8tL49IM8t/view?usp=sharing)
A dataset with mask labeling of three major types of concrete surface defects: crack, spalling and exposed rebar, was prepared for training and testing of the DIS-YOLO model. In this dataset, three open-domain datasets [1-3] are exploited and merged with a bridge inspection image dataset [4] from the Highways Department. To use the dataset, you need comply with the terms and conditions of using the images from the Highways Department, therefore please write a statement and e-mail it to: czhangbd@connect.ust.hk.

# Setup for training and testing
1. Download the pretrained weights [modified Yolov3 weights]( https://drive.google.com/drive/folders/1LDE08DwQaA79-lq7NKPU7ieb_DOh9VSk?usp=sharing) and put it into the folder “pretrained_weights”.
2. Change MODEL_PATH and relevant training parameters in config.py. 
3. Create “train”, “val” and “test” folders in the “data” folder, and put the downloaded data into each folder according to the format of "train_sample". Then, run “pre_process.py” to get “ground_truth_cache.pkl” for each folder. 
4. Train model using different transfer learning and fine-tuning strategies. To fine-tune specific layers, please adjust “lock = True/False” for relevant layers in yolo3_net_pos.py.
5. After training, run “calculate_test_map.py” to evaluate mask-level mAP and mIoU.

# Test results
The instance segmentation accuracy mAP is up to 80% computed using the mask-level IoU metric of 0.5. The test speed is around 0.1 s per image (576×576 pixel resolution) on a PC running Ubuntu with an Intel® CoreTM i7-7700 CPU and a GeForce GTX 1060 6GB GPU. The comparison of test samples between different models is as follows:

<div align=center><img src="https://github.com/ZHANGKEON/DIS-YOLO/blob/master/output/sample_result.jpg"/></div>

# Training on custom dataset
To apply the code to a custom dataset, you need do the following items:
1. Prepare raw images and labeled masks for train/val/test.
2. Download the pretrained [yolov3.weights]( https://pjreddie.com/darknet/yolo/) and convert it to a “.ckpt” file. Please note that the original yolov3.weights has been trained for 80 common object classes. For custom dataset with different number of classes, you can manually crop the last class prediction channel, or select the relevant layers for restoring weight in “train_yolo3_mask.py”.
3. Change settings in config.py, such as MODEL_PATH, CLASSES and ANCHORS, etc.

# Public datasets from:
1. Li S, Zhao X and Zhou G. Automatic pixel‐level multiple damage detection of concrete structure using fully convolutional network. Comput-Aided Civ Inf Eng 2019; 34: 616–634.
2. Yang L, Li B, Li W, et al. Semantic metric 3D reconstruction for concrete inspection. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, Salt Lake City, UT, 18–22 June 2018, pp.1624–1632.
3. Yang X, Li H, Yu Y, et al. Automatic pixel‐level crack detection and measurement using fully convolutional network. Comput-Aided Civ Inf Eng 2018; 33: 1090–1109.
4. Zhang C, Chang CC and Jamshidi M. Concrete bridge surface damage detection using a single‐stage detector. Comput-Aided Civ Inf Eng 2020; 35: 389–409.
# Citation
1. If you are using the dataset, please cite all papers that have contributed to the above dataset.
2. Zhang C, Chang C, Jamshidi M, Simultaneous Pixel-level Concrete Defect Detection and Grouping using a Fully Convolutional Model, Struct. Health Monit 2021.
