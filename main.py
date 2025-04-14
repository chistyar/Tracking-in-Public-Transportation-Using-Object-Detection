from imageai.Detection import ObjectDetection

detector = ObjectDetection()

detector.setModelTypeAsRetinaNet()

detector.setModelTypeAsYOLOv3()

detector.setModelPath("yolov3.pt") #choosing the model

detector.loadModel()

custom = detector.CustomObjects(person=True)

detections = detector.detectObjectsFromImage(
    input_image="peop1.jpg", output_image_path="1.jpg", minimum_percentage_probability=20) #uploading and getting back the img
[print(i) for i in (detections)] # printing info about all objects
print(sum([1 for i in detections if i["name"] == "person"]))#func for counting people in real time







