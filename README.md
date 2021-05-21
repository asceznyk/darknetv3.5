# oddnet (Object Detection Net, the extra d is for branding)

# Basic instructions

## 1. Clone the repo
```
$ git clone https://github.com/asceznyk/oddnet.git
```

## 2. Make directories
```
$ mkdir outputs/ gtruths/
```
- outputs is for storing model predictions
- gtruths is for stroing actual images with the ground truth boxes

## 3. Download pre-trained weights
- link for .weights file https://pjreddie.com/media/files/yolov3.weights

## 4. Download the .cfg file
- link for .cfg file https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg

## 5. Download pre-trained darknet53.conv.74 weights
- link for weights https://pjreddie.com/media/files/darknet53.conv.74

# How to use oddnet for: 
1. Plain Object Detection 
2. Custom Object Detection

# 1. Plain Object Detection

## Run the command
```
$ python detect.py --testdir yourtestdir --names yournames.labels --cfg yolov3.cfg --weights yolov3.weights --savedir outputs --boxdir gtruths
```
### Arguments:
- testdir: the directory with all the images to test on
- names: is a text file with all the names of classes in it like so:
```
class1
class2
class3
```
- cfg: is a model configuration file (this is for model architecture)
  * use the yolov3.cfg file you downloaded 
- weights: is a .pth or a .weights file you can get this from here
  * use the yolov3.weights file you downloaded
- savedir: Directory to save all the predictions of the model
- boxdir (optional):  Directory to save all the actual images wit ground truth boxes
