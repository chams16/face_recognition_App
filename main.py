import os.path
import numpy as np
import cv2
import json
from flask import Flask,request,Response
import uuid

#function detect face from image
def faceDetect(img):
    face_cascade = cv2.CascadeClassifier('face-detect-cascade.xml')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0))

    #save file
    path_file=('static/%s.jpg' %uuid.uuid4().hex)
    cv2.imwrite(path_file,img)
    return json.dumps(path_file) #retourne image file name

#API
app = Flask(import_name='__name__')

#route http post to this method
@app.route('/api/upload',methods=['POST'])
def upload():
    #retrieve image from client
    img = cv2.imdecode(np.fromstring(request.files['image'].read(),np.uint8),cv2.IMREAD_UNCHANGED)
    #process image
    img_processed = faceDetect(img)
    #response
    return Response(response=img_processed,status=200,mimetype="application/json") #return json string

#start server
app.run(host="0.0.0.0",port=5000)


