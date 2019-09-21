import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras import Model, optimizers
from keras import regularizers
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.optimizers import SGD, Adam

import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
import numpy as np
from sklearn.externals import joblib




IMG_W = 160
IMG_H = 160
IMG_CHNL = 3

# Encoders
in_encoder = Normalizer()
out_encoder = LabelEncoder()

# Label Encoding
labels = np.load('data/label-facenet.npy',allow_pickle=True)
out_encoder.fit(labels)

# SVM_MODEL LOAD FOR PREDICTION
model = joblib.load('models/facenet_svm_model_our_faces_added.sav')
facenet_model = load_model('models/facenet_keras.h5')



def get_embedding(face):
    # standardization
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face-mean)/std

    sample = np.expand_dims(face, axis=0)
    print(face.shape)
    # make prediction to get embedding
    yhat = facenet_model.predict(sample)
#     print(yhat[0].shape)
    return yhat[0]



def face_recognition(raw_img):
    # Read and process image for model
    # rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(raw_img, (IMG_W,IMG_H))

    # img = np.reshape(img, (1,img.shape[0], img.shape[1], img.shape[2]))
    img_emd = np.asarray(get_embedding(img)) # converting embedded image to numpy array if needed
    
#     print(img_emd.shape)
    img_emd = np.expand_dims(img_emd, axis=0)
    # Predicting Image
    img_norm = in_encoder.transform(img_emd)

    yhat_class = model.predict(img_norm)
    yhat_prob = model.predict_proba(img_norm)

    # Reverse Transform to Original label
    class_index = yhat_class[0]
    class_probability = yhat_prob[0,class_index] * 100
    predict_names = out_encoder.inverse_transform(yhat_class)

    return predict_names[0], class_probability
