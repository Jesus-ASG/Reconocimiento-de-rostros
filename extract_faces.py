import cv2
import os

inputPath = "user1/"
outputPath = 'user1_faces/'

if not os.path.exists(outputPath):
     os.makedirs(outputPath)

# Detector facial
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

for imageName in os.listdir(inputPath):
     image = cv2.imread(inputPath + "/" + imageName)
     faces = faceClassif.detectMultiScale(image, 1.1, 5)
     for (x, y, w, h) in faces:
          #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
          face = image[y:y + h, x:x + w]
          face = cv2.resize(face, (150, 150))
          cv2.imwrite(outputPath + imageName, face)