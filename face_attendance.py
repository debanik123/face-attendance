#!/usr/bin/env python
# coding: utf-8

# In[1]:



# Import OpenCV2 for image processing
import cv2,os
import sqlite3
import numpy as np
from PIL import Image 
import pickle
import datetime
import os 
import pandas as pd 
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()

assure_path_exists("trainer/")

# Load the trained mode
recognizer.read('trainer/trainer.yml')

# Load prebuilt model for Frontal Face
cascadePath = "haarcascade_frontalface_default.xml"

# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath);

# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize and start the video frame capture
cam = cv2.VideoCapture(0)            
Id_1= []
dt_1 = []
uid = []
# Loop
while True:
    # Read the video frame
    ret, im =cam.read()

    # Convert the captured frame into grayscale
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    # Get all face from the video frame
    faces = faceCascade.detectMultiScale(gray, 1.2,5)

    # For each face in faces
    for(x,y,w,h) in faces:

        # Create rectangle around the face
        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

        # Recognize the face belongs to which ID
        Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        print(Id)
        # Check the ID if exist 
        #if(profile!=None):
        if(confidence<85):
            if(Id == 1):
                Id = "Debanik {0:.2f}%".format(round(100 - confidence, 2))
                Id_name="Debanik"
                Id_1.append(Id_name)
                uid.append("TNU2017025100001")
            if(Id == 2):
                Id = "Payel {0:.2f}%".format(round(100 - confidence, 2)) 
                Id_name = "Payel"
                Id_1.append(Id_name)
                uid.append("TNU2017067100003")
            if(Id == 3):
                Id = "Susovan {0:.2f}%".format(round(100 - confidence, 2)) 
                Id_name = "Susovan"
                Id_1.append(Id_name)
                uid.append("TNU2017025100015")
        else:
            Id = "unknown_person {0:.2f}%".format(round(100 - confidence, 2)) 
            Id_name = "unknown_person"
            Id_1.append(Id_name)
            uid.append("Unknown_UID")
            
            
            
        dt = str(datetime.datetime.now())
        dt_1.append(dt)
        # Put text describe who is in the picture
        
        
        cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
        cv2.putText(im, str(Id)+dt, (x,y-40), font, 1, (255,255,255), 3)
        if Id == "unknown_person {0:.2f}%".format(round(100 - confidence, 2)):
            cv2.imwrite('unknown_person.png',cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1))
        
            

            
    
    # Display the video frame with the bounded rectangle
    cv2.imshow('im',im) 
    
    # If 'q' is pressed, close program
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
# Stop the camera
cam.release()

# Close all windows
cv2.destroyAllWindows()
dict = {"uid":uid,"name":Id_1,"date_time":dt_1}
df = pd.DataFrame(dict)
df.to_csv("name_face.csv")
new_f = df.drop_duplicates(subset='name', keep="first")
new_l = df.drop_duplicates(subset='name', keep="last")
frame = pd.concat([new_f,new_l])
frame.to_csv("name_face_clean.csv")


# In[55]:


def send_mail(sender,receiver):
    import smtplib 
    from email.mime.multipart import MIMEMultipart 
    from email.mime.text import MIMEText 
    from email.mime.base import MIMEBase 
    from email import encoders 

    fromaddr = sender
    toaddr = receiver
    msg = MIMEMultipart() 
    msg['From'] = fromaddr 
    msg['To'] = toaddr 
    msg['Subject'] = "Unknown_Person"
    body = "Unknown_Person"
    msg.attach(MIMEText(body, 'plain')) 
    filename = "File_name_with_extension"
    attachment = open("unknown_person.png", "rb") 
    p = MIMEBase('application', 'octet-stream') 
    p.set_payload((attachment).read()) 
    encoders.encode_base64(p) 
    p.add_header('Content-Disposition', "attachment; filename= %s" % filename) 
    msg.attach(p) 
    s = smtplib.SMTP('smtp.gmail.com', 587) 
    s.starttls() 
    s.login(fromaddr, "Djdeba@123") 
    text = msg.as_string() 
    s.sendmail(fromaddr, toaddr, text)  
    s.quit() 


# In[56]:


send_mail("debanikroy92@gmail.com","debanikroy.in@gmail.com")


# In[ ]:




