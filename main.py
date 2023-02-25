#Creating function for directories needed for image storing
import os
import os.path
import cv2

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        return None
    else :
        pass

# DataSet Building
cap = cv2.VideoCapture(0)

i=0
image_Count=0

while i < 10:

    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    #region of intereset
    x = 100
    y = 400
    w = 320
    z = 620
    roi = frame[x:y, w:z]
    cv2.imshow('roi',roi)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi , (28, 28) ,interpolation = cv2.INTER_AREA)

    cv2.imshow('roi sacled and gray ',roi)
    copy = frame.copy()
    cv2.rectangle(copy,(320, 100) ,(620, 400) ,(255,0,0), 5)

    if i == 0:
        image_Count = 0
        cv2.putText(copy, " Hit Enter to Record when Ready " ,(100,100) , cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,0), 1)

    if i == 1:
        image_Count += 1
        cv2.putText(copy, " Recording 1st Gesture -Train ", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        cv2.putText(copy,str(image_Count) , (400, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        gesture_one = './handgestures/train/0/'
        make_dir(gesture_one)
        cv2.imwrite(gesture_one + str(image_Count) + ".jpg" ,roi)

    if i == 2:
        image_Count += 1
        cv2.putText(copy, " Recording 1st Gesture -Train ", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        cv2.putText(copy, str(image_Count), (400, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        gesture_one = './handgestures/test/0/'
        make_dir(gesture_one)
        cv2.imwrite(gesture_one + str(image_Count) + ".jpg", roi)

    if i == 3:
        cv2.putText(copy, " Hit Enter to Record when Ready to recond 2nd gesture " ,(100,100) , cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,0), 1)

    if i == 4:
        image_Count += 1
        cv2.putText(copy, " Recording 2nd Gesture -Train ", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        cv2.putText(copy,str(image_Count) , (400, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        gesture_two = './handgestures/train/1/'
        make_dir(gesture_two)
        cv2.imwrite(gesture_two + str(image_Count) + ".jpg" ,roi)

    if i == 5:
        image_Count += 1
        cv2.putText(copy, " Recording 2nd Gesture -Train ", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        cv2.putText(copy, str(image_Count), (400, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        gesture_two = './handgestures/test/1/'
        make_dir(gesture_two)
        cv2.imwrite(gesture_two + str(image_Count) + ".jpg", roi)

    if i == 6:
        cv2.putText(copy, " Hit Enter to Record when Ready to recond 3rd gesture " ,(100,100) , cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,0), 1)

    if i == 7:
        image_Count += 1
        cv2.putText(copy, " Recording 3rd Gesture -Train ", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        cv2.putText(copy,str(image_Count) , (400, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        gesture_three = './handgestures/train/2/'
        make_dir(gesture_three)
        cv2.imwrite(gesture_three + str(image_Count) + ".jpg" ,roi)

    if i == 8:
        image_Count += 1
        cv2.putText(copy, " Recording 3rd Gesture -Train ", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        cv2.putText(copy, str(image_Count), (400, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        gesture_three = './handgestures/test/2/'
        make_dir(gesture_three)
        cv2.imwrite(gesture_three + str(image_Count) + ".jpg", roi)

    if i == 9:
        cv2.putText(copy, " Hit Enter to Exit ", (100, 100),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
    cv2.imshow('frame', copy)

    if cv2.waitKey(1) == 13 : #13 is Enter Key
        image_Count = 0
        i+=1

cap.release()
cv2.destroyAllWindows()

