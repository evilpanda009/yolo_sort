#track people using deep sort
import cv2
import numpy as np
import sort as s
import torch
mot_tracker = s.Sort()
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True) # for PIL/cv2/np inputs and NMS

model.classes = [0] # person class
def track_people(frame):
    preds = model(frame)
    print(preds.pandas().xyxy[0])
    detections = preds.pred[0].numpy()
    track_bbs_ids = mot_tracker.update(detections)
    for j in range(len(track_bbs_ids.tolist())):
        coords = track_bbs_ids.tolist()[j]
        cv2.rectangle(frame, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), (0, 255, 0), 2)
        cv2.putText(frame, str(int(coords[4])), (int(coords[0]), int(coords[1])), fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1,color= (0, 0, 255),thickness= 2)
        cv2.imshow("Image", frame)

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        track_people(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
