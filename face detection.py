import cv2

trainedDataset = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video = cv2.VideoCapture(0)

while True:
    Sucess, frame = video.read()
    if Sucess == True:
        grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = trainedDataset.detectMultiScale(grey_image)
        for x, y, w, h in face:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imshow('video', frame)
            cv2.waitKey(1)
    else:
        print("video Complate or Frame NILL")
        break
        