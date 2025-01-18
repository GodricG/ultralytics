from ultralytics import YOLO
import cv2 as cv
# Load a model
model = YOLO("weights/yolo11l.pt")

# Train the model
# train_results = model.train(
#     data="coco8.yaml",  # path to dataset YAML
#     epochs=100,  # number of training epochs
#     imgsz=640,  # training image size
#     device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
# )

# Evaluate model performance on the validation set
# metrics = model.val(data="ultralytics/cfg/datasets/coco.yaml")

# Perform object detection on an image
results = model(source="data/others")
for img in results:
    boxes = img.boxes.xywh.to("cpu").numpy()
    image = img.orig_img
    if len(boxes) != 0:
        x, y, w, h = boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3]
        x, y, w, h = int(x), int(y), int(w), int(h)
        x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
        print(x, y, w, h)
        image = cv.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
    image = cv.resize(image, (800, 600))
    cv.imshow("123", image)
    cv.waitKey()

# Export the model to ONNX format
# path = model.export(format="onnx")  # return path to exported model