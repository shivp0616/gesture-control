# Play game with your hand gestures

With the help of your webcam, play the T-Rex game (on google chrome).
We are using tensorflow api for the training of our machine learning model.

We are using CNN for the image prediction.

The packages and the dependencies that are used in this project are openCV, keras, tensorflow. Make sure that those are installed in your system.


First you have to train your model using the train.py file. Then run the main.py file

Before running the train.py, ensure that you have the dataset.
The structure for that should be like 
  
    ├── Dataset/
    │   ├── train/
    │   │   ├── class_A
    │   │   |   ├── img1
    │   │   |   ├── img2
    │   │   |   ├── img3
    │   │   ├── class_B   
    │   │   |   ├── img1
    │   │   |   ├── img2
    │   │   |   ├── img3
    │   ├── test/
    │   │   ├── class_A
    │   │   |   ├── img1
    │   │   |   ├── img2
    │   │   |   ├── img3
    │   │   ├── class_B
    │   │   |   ├── img1
    │   │   |   ├── img2
    │   │   |   ├── img3
    
    class_A & class_B are the two gestures which the model will learn
    
More the image, better the accuracy. One way to create the dataset is to start your webcam and click images of the hand gesture and then store it into particular folder.
