import cv2 as cv
import numpy as np

img_name = './images/minobia.jpg'

# nombres de las clases
with open('./input/coco_classes.txt','r') as f:
    class_names = f.read().split('\n')

COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

model = cv.dnn.readNet( model='./input/frozen_inference_graph.pb',
                        config='./input/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt',
                        framework='TensorFlow')

image = cv.imread(img_name)
image_height, image_width, _ = image.shape

blob = cv.dnn.blobFromImage(image=image, size=(300,300), mean=(104,117,123), swapRB=True)

model.setInput(blob)

output = model.forward()

#Loop over each of the detection
for detection in output[0, 0, :, :]:
    confidence = detection[2]
    if confidence >.3:
        class_id = detection[1]
        class_name = class_names[int(class_id)-1]
        color = COLORS[int(class_id)]
        box_x = detection[3] * image_width
        box_y = detection[4] * image_height
        box_width = detection[5] * image_width
        box_height = detection[6] * image_height
        cv.rectangle(image, (int(box_x), int(box_y)), (int(box_width),
                    int(box_height)), color, thickness=2)
        cv.putText(image, class_name, (int(box_x), int(box_y - 5)),
                    cv.FONT_HERSHEY_SIMPLEX, 3, color, 3)

cv.imshow('image', image)
cv.imwrite('image_result.jpg', image)
while(cv.waitKey(10) & 0xFF != ord('q')):
    a = 1
cv.waitKey(0)
cv.destroyAllWindows