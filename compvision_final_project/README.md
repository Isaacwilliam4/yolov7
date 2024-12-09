# Introduction

For my project, I am exploring ways to improve the YOLOv7 object detection model. 

---

# YOLOv7

The first phase of the project involved downloading the YOLOv7 source code and setting it up on my machine. This included:
- Downloading the COCO dataset, which contains over 100,000 images.
- Installing the required dependencies for YOLOv7.

I successfully got the real-time YOLOv7 model working with my webcam. This allowed me to observe YOLOv7's performance in real-time, analyzing how it classified objects as I introduced images into the frame. The YOLOv7 model I used was trained on 80 different object classes.

# Training Script
```
python3 train.py --workers 8 --device 0 --batch-size 32 --data data/coco.yaml --img 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolo_extra_training --hyp data/hyp.scratch.p5.yaml --epochs 25
```
