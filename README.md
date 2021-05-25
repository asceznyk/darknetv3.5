# oddnet (Object Detection Net, the extra d is for branding)

# Basic instructions

## 1. Update or install imgaug
```
$ pip uninstall -y imgaug && pip install git+https://github.com/aleju/imgaug.git
```

## 2. Clone the repo
```
$ git clone https://github.com/asceznyk/oddnet.git
```

## 3. Change directory to oddnet
```
$ cd oddnet/
```

## 4. Make directories
```
$ mkdir outputs/ gtruths/model
```
- outputs is for storing model predictions
- gtruths is for stroing actual images with the ground truth boxes

## 5. Download pre-trained weights
- link for .weights file https://pjreddie.com/media/files/yolov3.weights

## 6. Download the .cfg file
- link for .cfg file https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg

## 7. Download pre-trained darknet53.conv.74 weights (for custom object detection)
- link for weights https://pjreddie.com/media/files/darknet53.conv.74

# The format of dataset
1. images must be .jpg/.png files
2. create a .names file to store all the class names:
```
class1
class2
class3
```
each line in the .names file should be used for exactly 1 class
3. labels must be .txt files containing the boxes in follwing format:
```
c1 x y w h
c2 x y w h
```
- the x, y, w, h values must be between 0 and 1
- the c1, c2 values are the corresponding class indices depending on the .names file fog e.g:
```
class1 => c1 => 0
class2 => c2 => 1
```
the indices start at 0

# How to use oddnet for: 
1. Plain Object Detection 
2. Custom Object Detection

# Plain Object Detection

## Start detecting objects
```
$ python detect.py --testdir test --names yournames.labels --cfg yolov3.cfg --weights yolov3.weights --savedir outputs --boxdir gtruths
```
### Arguments:
- testdir: the directory with all the test images ONLY
- names: is a text file with all the names of classes in it like so:
- cfg: is a model configuration file (this is for model architecture)
  * use the yolov3.cfg file you downloaded 
- weights: is a .pth or a .weights file you can get this from here
  * use the yolov3.weights file you downloaded
- savedir: directory to save all the predictions of the model
- boxdir (optional): directory to save all the actual images wit ground truth boxes

# Custom Object Detection

## 1.  Create the custom model with number of classes
```
$ bash createmodel.sh nclasses
```
- here nclasses is the number of classes
- running this command will create a file called yolov3custom.cfg (this will be used in the next step)

## 2. Train the model
```
$ !python3 train.py --traindir train/ --validdir valid/ --cfg yolov3custom.cfg --ptweights darknet53.conv.74  --epochs 100 --ckptpth pathtochekpt.pth --lossfn bboxloss --patience 1000
```
### Arguments:
- traindir: the directory containing all the training images AND labels
- validdir: the directory containing all the validation images AND labels
- cfg: the custom model config file created by createmodel.sh (use the yolov3custom.cfg from the previous step)
- ptweights: pre-trained weights (use the darknet53.conv.74 weights downloaded, refer step 7 [here](https://github.com/asceznyk/oddnet/blob/main/README.md#7-download-pre-trained-darknet53conv74-weights-for-custom-object-detection)
- epochs: number of epochs to train the model
- ckptpth: checkpoint path, the path to save your model while training
- lossfn: loss function, you have two loss functions bboxloss and iouloss, it is best to use bboxloss because iouloss is broken (if you can find a creative way to fix, great! please ping me on asceznyk@gmail.com
- patience: this is the number of epochs to wait on stagnation of validation loss in order to stop training, the default value is 10 but you can change it to whatever you like

## 3. Detect custom objects
```
$ python detect.py --testdir test --names yournames.labels --cfg yolov3custom.cfg --weights pathtochekpt.pth --savedir outputs --boxdir gtruths 
```
### Arguments:
- testdir: the directory with all the test images ONLY
- names: is a text file with all the names of classes in it like so:
```
class1
class2
class3
```
- cfg: is a model configuration file (use the yolov3custom.cfg from the first step)
- weights: checkpoint path, the same checkpoint path from the previous step
- savedir: directory to save all the predictions of the model
- boxdir (optional): directory to save all the actual images wit ground truth boxes
