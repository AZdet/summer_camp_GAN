import cv2
import sys
import os
import pdb


CASCADE="Face_cascade.xml"
FACE_CASCADE=cv2.CascadeClassifier(CASCADE)

def detect_face(image_path):
    image=cv2.imread(image_path)
    if image is None:
        print('Load Image Error!')
        sys.exit(1)
    origin_h, origin_w, _ = image.shape
    image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(image_grey,scaleFactor=1.16,minNeighbors=5,minSize=(25,25),flags=0)
    if len(faces) == 0:
        return None
    #pdb.set_trace()
    # find the biggest face
    #largest_face = sorted(faces, cmp=lambda face1, face2: int(face1[2] * face1[3] - face2[2] * face2[3]))[-1]
    largest_face = sorted(faces, key=lambda face: int(face[2]*face[3]))[-1]

    x,y,w,h = largest_face
    bound = int(min(origin_h, origin_w) / 10)
    y_start = max(0, y - bound)
    x_start = max(0, x - bound)
    sub_img=image[y_start:y+h+bound,x_start:x+w+bound]
    sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2RGB)
    return sub_img




