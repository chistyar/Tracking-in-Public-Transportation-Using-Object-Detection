
import cv2
from imageai.Detection import ObjectDetection

detector = ObjectDetection()

detector.setModelTypeAsRetinaNet()

detector.setModelTypeAsYOLOv3()

detector.setModelPath("yolov3.pt")

detector.loadModel()

cam_feed = cv2.VideoCapture(1) #index of camera

while True:
    ret, img = cam_feed.read()
    annotated_image, preds = detector.detectObjectsFromImage(input_image=img,
                                                               
                                                               output_type="array",
                                                               display_percentage_probability=False,
                                                               display_object_name=True)#show boxes of objects
    print(sum([1 for i in preds if i["name"] == "person"])) #func for counting people in real time
    cv2.imshow("", annotated_image)

    if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1) == 27):
        break


cam_feed.release()

cv2.destroyAllWindows()


