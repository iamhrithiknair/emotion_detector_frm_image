from deepface import DeepFace
import cv2
import matplotlib.pyplot as p

img1 = cv2.imread('C:/Users/hrith/Downloads/rdj2.jpg')
p.imshow(img1[:,:,::-1])  #used to display the image represented by img1
p.show()                  #used to display the image window with the loaded image

res = DeepFace.analyze(img1, actions = ['emotion'])

print(res)