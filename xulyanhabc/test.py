from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os

model_path = 'yolov8x_trained.pt'
image_folder = 'dataset_XLA/test/images'

color_map = {
    'bicycle': (0, 255, 0),
    'bus': (255, 0, 0),
    'car': (0, 0, 255),
    'motorbike': (255, 255, 0),
    'person': (0, 255, 255)
}

model = YOLO(model_path)

for filename in os.listdir(image_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Không thể đọc ảnh từ đường dẫn: {img_path}")
            continue

        results = model(img, imgsz=640, augment=True, conf=0.1, iou=0.45)

        count = 0

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = result.names[int(box.cls[0])]
                confidence = box.conf[0]

                if label in color_map:
                    count += 1

                    color = color_map[label]
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
                    cv2.putText(
                        img,
                        f'{label} {confidence:.2f}',
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1
                    )

        print(f'{filename}: Total vehicles detected: {count}')

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 8))
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.title(f'{filename}: Total vehicles detected: {count}')
        plt.show()
