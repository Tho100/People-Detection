import numpy as np
import cv2
import matplotlib.pyplot as plt 

hog = cv2.HOGDescriptor()   
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cv2.startWindowThread()

img = cv2.imread("C:\\users\\USER\\Documents\\pedestrian2.png")
trainer = cv2.resize(img,(680,600))

boxes,weights = hog.detectMultiScale(trainer, winStride=(1,1))
boxes = np.array([[x,y,x+w,y+h] for (x,y,w,h) in boxes])

for xA,yA,xB,yB in boxes:
    cv2.rectangle(trainer,(xA,yA),(xB,yB),(255,0,255),2)    
    
fig = plt.figure()
fig.set_figwidth(10)
fig.set_figheight(8)

q = [img,trainer]
for i in range(len(q)):
    plt.subplot(1,2,i+1),plt.imshow(cv2.cvtColor(q[i],cv2.COLOR_BGR2RGB))  
    plt.title(f'Total Detected "Human": {len(boxes)}')
