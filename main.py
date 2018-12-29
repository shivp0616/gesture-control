# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 19:30:40 2018

@author: shivp1606
"""
#camera control
import cv2
import numpy as np
from PIL import Image
import pyautogui 

# Load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

from keras.models import model_from_json
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

video = cv2.VideoCapture(0)

while True:
        _, frame = video.read()

        #Convert the captured frame into RGB
        im = Image.fromarray(frame, 'RGB')

        #Resizing into 128x128 because we trained the model with this image size.
        im = im.resize((64,64))
        img_array = np.array(im)

        #Our keras model used a 4D tensor, (images x height x width x channel)
        #So changing dimension 128x128x3 into 1x128x128x3 
        img_array = np.expand_dims(img_array, axis=0)

        #Calling the predict method on model to predict 'me' on the image
        prediction = int(loaded_model.predict(img_array)[0][0])

        #if prediction is 0, which means I am missing on the image, then show the frame in gray color.
        if prediction == 0  :
                pyautogui.press('space')
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow("Capturing", frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
                break
video.release()
cv2.destroyAllWindows()

