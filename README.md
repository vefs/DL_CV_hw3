# DL_CV_hw3
Apply YOLOv3 approach for object detection
## Train
Training Mode
$  python -W ignore::UserWarning train.py --train=True
## Detect
Detection Mode
$  python -W ignore::UserWarning train.py --detect=True

## Traing dataset
Modify dataset_folder='data/yours' in train.py
Training Mode (ref data/digit_mini)
put imgage file to  'data/yours/images/train'
put label txt to    'data/yours/label/train'
create training file list 'data/yours/train.f
## Detect Mode 
only require put imgage file to  'data/yours/images/test'
  


