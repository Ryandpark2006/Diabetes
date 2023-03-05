import numpy as np
import cv2
import os 
from app import app, APP_ROOT
import tensorflow as tf
# import sklearn
# import pickle as pkl
# import glob
# import numpy as np
# import pandas as pd
# import os
# import librosa
# import librosa.display
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import joblib
# import mediapipe as mp
# import time
# import tensorflow
from keras.applications.resnet_v2 import preprocess_input
from scipy.special import softmax

temp_path = os.path.join(APP_ROOT, 'temp')

def spiral():
    # for path in os.listdir(temp_path):
    img_path = os.path.join(temp_path, 'spiral.png')
    
    # img_path = path
    # print(path)
    img = cv2.imread(img_path)
# plt.imshow(np.asarray(Image.open(img_path)))
    img = cv2.resize(img, dsize=(300, 300), interpolation=cv2.INTER_CUBIC)
    img=np.array(img)
    img = preprocess_input(img)

    images_list = []
    images_list.append(np.array(img))
    x = np.asarray(images_list)
    # model = tf.keras.models.load_model(os.path.join(APP_ROOT, 'lymemodel.h5'))
    # preds = model.predict([x])
    interpreter = tf.lite.Interpreter(model_path = os.path.join(APP_ROOT, 'LymeMobileQuant.tflite'))
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # interpreter.resize_tensor_input(input_details[0]['index'], (1, 300, 300))
    # interpreter.resize_tensor_input(output_details[0]['index'], (-1, 2))

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    tflite_model_predictions = interpreter.get_tensor(output_details[0]['index'])
    print("Prediction results shape:", tflite_model_predictions.shape)
    # prediction_classes = np.argmax(tflite_model_predictions, axis=1)

    preds = softmax(tflite_model_predictions, axis = 1)
    # pred = np.argmax(preds[0], axis=0)
    # prob = preds[0][pred]
    # return round(preds[0][1].item() * 100, 2)  # percent positive
    print(preds[0][0])
    return [round(preds[0][1].item() * 100, 2), img_path]  # percent positive
