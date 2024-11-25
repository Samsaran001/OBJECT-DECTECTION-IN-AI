import time
from ultralytics import YOLO
import cv2
import math
import datetime as dt
import itertools
import threading



cap=cv2.VideoCapture(0)

frame_rate = 30
delay_time = 1/frame_rate
#ds=dt.time(0,0,30)
frame_width=int(cap.get(3))
frame_height = int(cap.get(4))

out=cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'),10,(frame_width, frame_height))

model=YOLO("../YOLO-Weights/ppe-50.pt")
classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']
last_check_time = time.time()
now=dt.datetime.now().strftime("%Y%m%d_%H%M%S")
No_hard_hat_count = 0
No_safety_vest_count=0

detected_objects = {}

while True:
    success,img=cap.read()
    results=model(img,stream=True)
    #save_image=False
    current_time = time.time() 
    for r in results :
         boxes=r.boxes
         for box in boxes:
           x1,y1,x2,y2 = box.xyxy[0]
           print(x1,y1,x2,y2)
           x1,y1,x2,y2=int(x1) , int(y1), int(x2), int(y2)
           print(x1,y1,x2,y2)
           cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),1)
           print(box.conf[0])
           conf=math.ceil((box.conf[0]*100))/100
           cls=int(box.cls[0])
           class_name=classNames[cls]
           label = f'{class_name}{conf}'
           object_key = (x1, y1, x2, y2)
           detected = object_key in detected_objects
           s = 'NO-Hardhat'
           if s in label:
               cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),1)
           h = 'NO-Safety Vest'
           if h in label:
               cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
           t_size = cv2.getTextSize(label,0,fontScale=1,thickness=2)[0]
           print(t_size)
           c2 = x1 + t_size[0],y1 - t_size[1] - 3
           cv2.rectangle(img,(x1,y1),c2,[255,0,255],-1,cv2.LINE_AA)
           cv2.putText(img, label , (x1,y1-2) ,0 ,1,[255,255,255],thickness=1,lineType=cv2.LINE_AA)
           #now = time.localtime()
                      
           a='No-Hardhat'
           for a in label:
            save_image = True
            No_hard_hat_count += 1            
            if a == 'No-Hardhat':        
                        continue
                        #pass
            detected_objects[object_key] = current_time              
            if (current_time - last_check_time) >= 30:                  
                filename = f"13_{No_hard_hat_count}{now}.jpg"
                cv2.imwrite(filename, img)
                #last_check_time = current_time
                #print(detected_persons)
            if No_hard_hat_count == 0:
                    print('The "No-Hardhat" condition was met for all labels; no images were saved.')
           
           b='NO-Safety Vest' 
           for b in label:
            save_image=True
            No_safety_vest_count+=1                
            if b == 'NO-Safety Vest':
                       continue
                       #pass
            detected_objects[object_key] = current_time           
            if (current_time - last_check_time) >= 30:             
             
                filename=f"14_{No_safety_vest_count}{now}.jpg"
                cv2.imwrite(filename,img)
           
       
            if No_safety_vest_count == 0:                               
                    print('The "No-Hardhat" condition was met for all labels; no images were saved')

           
           last_check_time = current_time
    
    detected_objects = {k: v for k, v in detected_objects.items() if current_time - v <= 30}

    out.write(img)

    cv2.imshow("images",img)

    if cv2.waitKey(1) & 0xFF==ord('1'):
            break
cap.release()
out.release()
cv2.destroyAllWindows()