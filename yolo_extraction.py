#!pip install ultralytics
#!pip install opencv-python

import math
from ultralytics import YOLO
import cv2
from google.colab.patches import cv2_imshow
import numpy as np

#!pip install ultralytics
#!pip install opencv-python

def yolo8_bounding_boxes_with_distance(image_path):
    model = YOLO("yolov8l.pt")
    frame = cv2.imread(image_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    bboxes = model.predict(frame)
    result = bboxes[0]

    img_height, img_width, _ = frame.shape

    bounding_boxes = []
    box_sizes = []
    #confidence = []
    i = 0
    for box in result.boxes:
        #class_id = result.names[box.cls[0].item()]
        #print(box)
        class_id = box.cls[0].item()
        cords = box.xyxy[0].tolist()
        x1, y1, x2, y2  = [round(x) for x in cords]
        box_width = round(x2 - x1, 4)
        box_height = round(y2 - y1, 4)
        total_box = round(box_height * box_width, 4)
        cX, cY = x1 + int((x2-x1)/2), y1 + int((y2-y1)/2) # center X and center Y
        conf = box.conf[0].item()

        #print(class_id)
        #print("Object type:", class_id)
        #print("Coordinates:",  x1, y1, x2, y2)
        #print("Box width, height and total box:",  box_width, box_height, total_box )
        #print("Probability:", conf)
        #print("---")

        cv2.rectangle(frame, (x1, y1),(x2, y2), 2)
        bounding_boxes.append([
          x1, y1, x2, y2, class_id, conf, total_box, cX, cY
        ])
        #confidence.append(conf)
        box_sizes.append(total_box)
        i=i+1

    #cv2_imshow(frame)

    #print('bounding boxes : ', bounding_boxes)
    #print('box sizes : ', box_sizes)
    #print('confidences : ', confidence)
    if len(bounding_boxes) > 0 :
        update_bbox = {}
        distance = []

        # check whether multiple biggest bounding box sizes
        check_multi_box = []
        for i in range(len(bounding_boxes)):
            if bounding_boxes[i][6] == max(box_sizes):
                check_multi_box.append(i)
        #print('biggest bounding boxes index number : ', check_multi_box)
        # get one biggest bounding box according to confidence score
        if(len(check_multi_box) > 1) :
            for i in range(len(check_multi_box)):
                if i == 0 :
                    big_conf_score = bounding_boxes[check_multi_box[0]][5]
                else :
                    if bounding_boxes[check_multi_box[i]][5] > big_conf_score :
                        big_conf_score = bounding_boxes[check_multi_box[i]][5]
        else :
              big_conf_score = bounding_boxes[check_multi_box[0]][5]

        for bbox in bounding_boxes:
            if bbox[6] == max(box_sizes) and big_conf_score == bbox[5] :
                source_box = bbox
                #print("Biggest bousing box : ", source_box)
                for i in range(len(bounding_boxes)):
                    if source_box[7] != bounding_boxes[i][7] and source_box[8] != bounding_boxes[i][8] :

                        dist = math.sqrt((source_box[7] - bounding_boxes[i][7])** 2 + (source_box[8] - bounding_boxes[i][8])** 2)
                        update_bbox[i] = {index: element for index, element in enumerate(bounding_boxes[i])}
                        update_bbox[i][len(update_bbox[i])] = round(dist, 4)
                        distance.append(round(dist, 4))
                        #distance[i] = round(dist, 4)
                        #print("distance", round(dist, 4))
                    else:
                        update_bbox[i] = {index: element for index, element in enumerate(bounding_boxes[i])}
                        update_bbox[i][len(update_bbox[i])] = 0
                        distance.append(0)
                        #distance[i] = 0
                        #print("distance", 0)

        #print(len(update_bbox))
        #for i in range(len(update_bbox)):
        #    distance[i] = update_bbox[i][9]

        #print(update_bbox)
        #print(bboxes)

        #print("distance values : ", distance)
        #print("Sorted distance values : ", sorted(distance))
        #print("Sorted distance length : ", len(sorted(distance)))



        final_bbox = []

        # To eleminate multiple entry of same element make an array
        eleminate_arr = []
        for i in range(0, len(sorted(distance))) :
            for j in range(0, len(sorted(distance))) :
                if sorted(distance)[i] == update_bbox[j][9] and j not in eleminate_arr :
                    del update_bbox[j][8], update_bbox[j][7], update_bbox[j][6]
                    #print("updated box : ", update_bbox[j].values())
                    final_bbox.append(list(update_bbox[j].values()))
                    eleminate_arr.append(j)
                #break

        #for i in range(len(sorted(distance))):
        #    for key, val in update_bbox.items():
                #print(val)
        #        if sorted(distance)[i] == val[9] :
                    #print(val)
                    #del val[8], val[7], val[6]
                    #print(val)
                    #del val[8]
                    #del val[7]
                    #del val[6]
                    #val.pop(8)
                    #val.pop(7)
                    #val.pop(6)
        #           final_bbox.append(list(val.values()))
                    #print(list(val.values()))


        #print(final_bbox)
        #print(np.array(final_bbox))

        return np.array(final_bbox)
    else :
        #return np.empty([1, 7])
        return np.array([0 for x in range(7)])

#final_bbox = yolo8_bounding_boxes_with_distance("/content/train2014/COCO_train2014_000000001997.jpg") # TEST ONLY
#final_bbox = yolo8_bounding_boxes_with_distance("/content/babyCatchingFish.jpg")  # TEST ONLY
#print(final_bbox)