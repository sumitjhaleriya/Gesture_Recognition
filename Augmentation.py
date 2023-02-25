import tensorflow
import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
import classifier
import scipy
import image
import os
import cv2


num_classes = 3
img_rows,img_cols = 28,28
batch_size = 32

train_data_dir ='./handgestures/train'
validation_data_dir ='./handgestures/test'

# data augmentation
train_data_genern = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_data_genern = ImageDataGenerator(rescale=1./255)

train_generator = train_data_genern.flow_from_directory(
    train_data_dir,
    target_size =(img_rows,img_cols),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    #classes=range(3)
)

validation_generator = val_data_genern.flow_from_directory(
    validation_data_dir,
    target_size =(img_rows,img_cols),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    #classes=range(3)
)

#
# #Model Creation
model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), activation= 'relu', input_shape=(28,28,1) ))
model.add(MaxPooling2D(pool_size=(2,2) ))

model.add(Conv2D(64, kernel_size=(3,3), activation= 'relu' ))
model.add(MaxPooling2D(pool_size=(2,2) ))

model.add(Conv2D(64, kernel_size=(3,3), activation= 'relu' ))
model.add(MaxPooling2D(pool_size=(2,2) ))

model.add(Flatten())
### For multiple signs

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.20))
model.add(Dense(3, activation='softmax')) ##3 --> is the numbers of classesss

###

#For 2 signs
# model.add(Dense(128, activation= 'relu'))
# model.add(Dropout(0.20))
# #1<->num_Classes
# model.add(Dense(1, activation= 'sigmoid'))

###

print(model.summary())

#Training the model
model.compile(loss ='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

nb_train_samples = 2887
nb_validation_samples = 1061
epochs = 10

history = model.fit(
    train_generator ,
    steps_per_epoch= nb_train_samples // batch_size,
    epochs=epochs,
    validation_data= validation_generator,
    validation_steps =nb_validation_samples // batch_size
)

#Saving the Model
model.save("My_Gesture_CNN.h5")

classifier = load_model('My_Gesture_CNN.h5')


#Testing the Model

cap = cv2.VideoCapture(0)

while True:

    ret, frame =cap.read()
    frame = cv2.flip(frame,1)

    #defining roi
    # region of intereset
    x = 100
    y = 800
    w = 320
    z = 920
    roi = frame[x:y, w:z]
    cv2.imshow('roi', roi)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    cv2.imshow('roi scaled and gray ', roi)
    copy = frame.copy()
    cv2.rectangle(copy, (320, 100), (620, 400), (255, 0, 0), 5)

    roi = roi.reshape(1,28,28,1)
    roi = roi/255
    ##For 2 Signs :
    #y_predict = (classifier.predict(roi) > 0.5).astype("int32")

    class_names = {0: 'scissor', 1: 'paper', 2: 'stone'}
    y_predict = classifier.predict(roi)
    predicted_class_index = np.argmax(y_predict, axis=1)
    predicted_class_names = [class_names[i] for i in predicted_class_index]

    # # Get the predicted class probabilities
    # y_predict = classifier.predict(roi)
    #
    # # Get the index of the class with the highest probability
    # predicted_class = np.argmax(y_predict)
    #
    # # Convert the index to the actual class label using the class_indices attribute of the generator
    # class_labels = list(train_generator.class_indices.keys())
    # predicted_label = class_labels[predicted_class]
    #
    # print("Predicted Class Set : ", class_labels)
    # # Print the predicted label
    # print("Predicted class printed :", predicted_label)
    #y_map = {0: 'cat', 1: 'dog', 2: 'bird'}
    #print(y_map[predicted_label])
    #y_pred_mapped = [label_map[i] for i in class_labels]
    #result = str(y_predict[0][0])

    #Printing mapped
    cv2.putText(copy,str(predicted_class_names[0]),(300,400),cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
    cv2.imshow('frame',copy)

    if cv2.waitKey(1) == 13:  # 13 is Enter Key
        break

cap.release()
cv2.destroyAllWindows()


