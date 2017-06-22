import numpy as np
import os
import cv2

def load_bbox(ground_file,resize,dataformat=0):
    f = open(ground_file)
    lines=f.readlines()
    bbox=[]
    for line in lines:
        if line:
            pt= line.strip().split(',')
            bbox.append(pt)
    bbox = np.array(bbox)
    if resize:
         bbox = (bbox.astype('float32')/2).astype('int')
    else:
        bbox = bbox.astype('float32').astype('int')
    if dataformat:
        bbox[:,2] = bbox[:,0]+bbox[:,2]
        bbox[:,3] = bbox[:,1]+bbox[:,3]
    return bbox

def load_imglst(img_dir):
    file_lst = [pic for pic in os.listdir(img_dir) if '.jpg' in pic]
    img_lst = [os.path.join(img_dir,filename) for filename in file_lst]
    return img_lst 


def display_tracker(img_lst,bbox_lst,save_flag):
    length = min(len(img_lst),len(bbox_lst))
    for i in range(length):
        img = cv2.imread(img_lst[i])
        visual(img,bbox_lst[i])
        if save_flag:
            if i%50==0:
                save(img,bbox_lst[i],str(i)+'.png')
    cv2.destroyAllWindows()

def visual(img,bbox):
    (x,y,w,h) = bbox
    pt1,pt2 = (x,y),(x+w,y+h)
    img_rec = cv2.rectangle(img,pt1,pt2,(0,255,255),2)
    cv2.imshow('window',img_rec)
    cv2.waitKey(10)

def save(img,bbox,name):
    (x,y,w,h) = bbox
    pt1,pt2 = (x,y),(x+w,y+h)
    img_rec = cv2.rectangle(img,pt1,pt2,(0,255,255),2)
    img_out = cv2.resize(img_rec,(150,100))
    cv2.imwrite(str(w)+name,img_out)