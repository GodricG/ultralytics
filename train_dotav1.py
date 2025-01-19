# from ultralytics import YOLO
from ultralytics.models.yolo import YOLO
import cv2 as cv

# Create a new YOLO11n-OBB model from scratch
# model = YOLO("yolo11m-obb.yaml").load("runs/obb/train2/weights/best.pt")
model = YOLO("yolo11m-obb.yaml").load("yolo11m-obb.pt")

# Costumed model
# model = YOLO("my_cfg/yolo11-obb-c.yaml").load("yolo11n-obb.pt")

# Train the model on the DOTAv1 dataset
# results = model.train(data="DOTAv1.yaml", epochs=2, imgsz=512, workers=0)

results = model("data/DOTAv1/images/test/P0014.jpg")

# def draw_rectangle(image, list_xy):
#     for xy in list_xy:
#
#
#     return img

for result in results:
    obb = result.obb
    # image = result.orig_img
    # if len(obb.xyxyxyxy) != 0:
    #     xyxyxyxy = obb.xyxyxyxy.to("cpu").numpy()
    #     x1, y1, x2, y2, x3, y3, x4, y4 = int(x1), int(y1), int(x2), int(y2), int(x3), int(y3), int(x4), int(y4)
    #     image = cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
    #     image = cv.line(image, (x2, y2), (x3, y3), (0, 0, 255), 1)
    #     image = cv.line(image, (x3, y3), (x4, y4), (0, 0, 255), 1)
    #     image = cv.line(image, (x4, y4), (x1, y1), (0, 0, 255), 1)
    # image = cv.resize(image, (800, 600))
    # cv.imshow("result", image)
    # cv.waitKey()
    result.show()
    result.save(filename="orig.jpg")