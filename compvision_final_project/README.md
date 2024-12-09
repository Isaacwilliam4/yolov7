# Introduction

For my project, I am exploring ways to improve the YOLOv7 object detection model. 
---

# YOLOv7

The first phase of the project involved downloading the YOLOv7 source code and setting it up on my machine. This included:
- Downloading the COCO dataset, which contains over 100,000 images.
- Installing the required dependencies for YOLOv7.

I successfully got the real-time YOLOv7 model working with my webcam. This allowed me to observe YOLOv7's performance in real-time, analyzing how it classified objects as I introduced images into the frame. The YOLOv7 model I used was trained on 80 different object classes.

# Setup

In order to install the appropriate python packages, run the following 

```
cd yolov7/
pip install -r requirements.txt
```

# Training Script
```
python3 train.py --workers 8 --device 0 --batch-size 32 --data data/coco.yaml --img 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolo_extra_training --hyp data/hyp.scratch.p5.yaml --epochs 25
```

# Notebooks
I used notebooks to facilitate easier debugging while developing the code for the various tasks. Each function in the notebook contains a clear description of its purpose, below I outline a general description of each notebook.

- notebooks/mean_avg_pooling.ipynb
    This notebook shows my exploration/curation of the method that applies the shrinking function to the image, and ensures that the corresponding bounding box labels still match up.
- notebooks/size_dist.ipyng
    This notebook shows the resulting analysis of applying the shrinking method on the coco dataset and the resulting average confidence score for each different bounding box size. 
- notebooks/add_small_imgs.ipynb
    This notebooks contains the functions that shrink the images down, add them to the new training dataset, and update the corresponding labels to ensure training works.
