import cv2
import os

# Create the images directory if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')

video = cv2.VideoCapture(0)  # Input device

# Provide the correct paths to the Haarcascade XML files
faceCascade = cv2.CascadeClassifier(r"D:\autocapture selfie\haarcascadefacial.xml")
smileCascade = cv2.CascadeClassifier(r"D:\autocapture selfie\haarcascdesmile.xml")

# Check if cascades are loaded successfully
if faceCascade.empty():
    print("Error loading face cascade")
if smileCascade.empty():
    print("Error loading smile cascade")

while True:
    success, img = video.read()
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
    faces = faceCascade.detectMultiScale(grayImg, 1.1, 4)
    cnt = 500
    keyPressed = cv2.waitKey(1)
    for x, y, w, h in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 3)
        smiles = smileCascade.detectMultiScale(grayImg, 1.8, 15)
        for x, y, w, h in smiles:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (100, 100, 100), 5)
            path = r'images\img' + str(cnt) + '.jpg'
            print(f"Saving image to: {path}")
            cv2.imwrite(path, img)
            print("Image " + str(cnt) + " Saved")
            cnt += 1
            if cnt >= 503:
                break

    cv2.imshow('live video', img)
    if keyPressed & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
