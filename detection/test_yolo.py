from pathlib import Path
from ultralytics import YOLO

# -------------------------
# settings
# -------------------------
CKPT = ""
DATA_YAML = ""

IMGSZ = 640
SPLIT = "test"     
DEVICE = 1          
BATCH = 1

CONF = 0.001
IOU_NMS = 0.7
MAX_DET = 100

def main():
    ckpt_path = Path(CKPT)
    data_path = Path(DATA_YAML)

    model = YOLO(str(ckpt_path))

    metrics = model.val(
        data=str(data_path),
        imgsz=IMGSZ,
        split=SPLIT,
        batch=BATCH,
        device=DEVICE,
        conf=CONF,
        iou=IOU_NMS,
        max_det=MAX_DET,
        save_json=True,
        save_txt=True,
        save_conf=True,
        plots=False,  
        verbose=False
    )

    print("=== Ultralytics DetMetrics ===")
    print(f"mAP50-95:  {float(metrics.box.map):.3f}")


if __name__ == "__main__":
    main()