# Play game with your hand gestures

With the help of your webcam, play the T-Rex game (on google chrome).
We are using tensorflow api for the training of our machine learning model.

We are using CNN for the image prediction.

To access the webcam, we are using opencv.

First you have to train your model using the train.py file. Then run the main.py file

Before running the train.py, ensure that you have the dataset.
The structure for that should be like 
Dataset
  |
  |-train
  | 
  |train test class_A class_B
  
  
    ├── Dataset/
    │   ├── train/
    │   │   ├── class_A
    │   │   ├── class_B   
    │   ├── test/
    │   │   ├── class_A
    │   │   ├── class_B

