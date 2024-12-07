from ultralytics import YOLO

data_path = '/home/thienbd/Desktop/xulyanh/data.yaml'

model = YOLO('yolov8x.pt')

model.train(
    data = data_path,
    epochs = 50,
    imgsz = 640,
    batch = 16,
    workers = 8,
    optimizer = 'AdamW',
    device = 0,
    patience = 10,
    amp =True
)

# yolo train model=/home/thienbd/Desktop/xulyanh/runs/detect/train2/weights/best.pt data=/home/thienbd/Desktop/xulyanh/data.yaml epochs=10 mosaic=0 lr0=0.0005
# yolo train model=/home/thienbd/Desktop/xulyanh/runs/detect/train/weights/best.pt data=/home/thienbd/Desktop/xulyanh/data.yaml epochs=50
