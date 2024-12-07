from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model_path = 'runs/detect/train/weights/best.pt'
image_path = 'C:/Users/ntrgi/Desktop/xulyanh/image.png'

color_map = {
    'bicycle': (0, 255, 0),
    'bus': (255, 0, 0),
    'car': (0, 0, 255),
    'motorbike': (255, 255, 0),
    'person': (0, 255, 255)
}

model = YOLO(model_path)

img = cv2.imread(image_path)

if img is None:
    print(f"Không thể đọc ảnh từ đường dẫn: {image_path}")
else:
    results = model(img, augment=True, conf=0.5, iou=0.5)

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

    print(f'Total vehicles detected: {count}')

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title(f'Total vehicles detected: {count}')
    plt.show()