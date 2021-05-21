# oddnet (Object Detection Net, extra d for branding)

# How to use oddnet for: 
1. Plain Object Detection 
2. Custom Object Detection

# Plain Object Detection 

1. Clone the repo
```
$ git clone https://github.com/asceznyk/oddnet.git
```

2. Make directories
```
$ mkdir outputs/ gtruths/
```
outputs is for storing model predictions
gtruths is for stroing actual images with the ground truth boxes

3. Run the command
```
$ python detect.py --testdir test --names test/names.labels --cfg model.cfg --weights custom.pth --savedir outputs/ --boxdir gtruths/ 
```
Arguments:
1. testdir: the directory with all the images to test on
2. names: is a text file with all the names of classes in it like so
3. an example .labels or .names file:
```
class1
class2
class3
```
4.cfg: is a model configuration file (this is for model architecture)
example .cfg files
https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
https://github.com/pjreddie/darknet/blob/master/cfg/yolov1.cfg
5.weights: is a .pth or a .weights file you can get this from here:
https://pjreddie.com/media/files/yolov3.weights
6.savedir: Directory to save all the predictions of the model
7.boxdir (optional):  Directory to save all the actual images wit ground truth boxes
