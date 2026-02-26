from ultralytics import YOLO

# load pretrained YOLOv9e model from Ultralytics
model = YOLO("yolov9e.pt") 

# ============================================
# Dataset structure (Ultralytics YOLO format)
# ============================================
# data.yaml example:
# path: /dataset_root
# train: train/images
# val: valid/images
# test: test/images
# nc: 4  # number of classes
# names: ['echinus', 'holothurian', 'scallop', 'starfish']
#
# NOTE:
# Ultralytics resolves label paths automatically by
# replacing 'images' with 'labels' in the directory path.
#
# ============================================
# Required directory layout:
#
# dataset_root/
# ├── train/
# │   ├── images/*.jpg
# │   └── labels/*.txt
# │
# ├── valid/
# │   ├── images/*.jpg
# │   └── labels/*.txt
# │
# └── test/
#     ├── images/*.jpg
#     └── labels/*.txt
#
# Additional folders (optional and ignored by YOLO):
# - Annotations/ : VOC XML annotations
# - gt/          : enhancement ground truth images
#
# ============================================
# Label format (YOLO):
# <class_id> <cx> <cy> <w> <h>
# - normalized coordinates in range [0, 1]
# - multiple objects are stored as multiple lines
# ============================================

data_yaml = "./data/data.yaml"
run_name  = ""


model.train(
    data=data_yaml,
    epochs=200,
    imgsz=640,
    batch=16,
    name=run_name,
    workers=8,
    resume=False,
    device=[0],

    # ---- For fair comparison on the image quality, all data augmentation is turned OFF ----
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    degrees=0.0,
    translate=0.0,
    scale=0.0,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.0,
    mosaic=0.0,
    mixup=0.0,
    cutmix=0.0,
    copy_paste=0.0,
    auto_augment="none",   
    erasing=0.0,          
)
