# python select_roi.py --cam <cam_url> --num <num> 
# --cam: camera url
# --num: number of points in roi. [optional, default: 4]
import numpy as np
import cv2
from matplotlib import pyplot as plt
import argparse


ap = argparse.ArgumentParser()
ap.add_argument('--cam', help='Image to select points')
ap.add_argument('--num', type=int, default=4, help='No. of points')
args = ap.parse_args()

cam_url = args.cam
if cam_url in ['0', '1']:
    cam_url = int(cam_url)

cap = cv2.VideoCapture(cam_url)
frame = cap.read()[1]
cap.release()

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
plt.imshow(frame)
pts = plt.ginput(args.num)
plt.show()
plt.close()

pts = np.array(pts, dtype=int)
pts_list = [[i[0], i[1]] for i in pts]
print(pts_list)

cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

plt.imshow(frame)
plt.show()
