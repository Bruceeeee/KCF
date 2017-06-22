import os
import numpy as np
import cv2 
import HOG



def crop_img(img,bbox):
    (x,y,w,h) = bbox
    return img[x:x+w,y:y+h]

def generate_gaussian(win_size,sigma):
    h,w = win_size
    rx = np.arange(w/2)
    ry = np.arange(h/2)
    x = np.hstack((rx,rx[::-1]))
    y = np.hstack((ry,ry[::-1]))
    xx,yy = np.meshgrid(x,y)
    y_reg = np.exp(-1*(xx**2+yy**2)/(sigma**2))
    return y_reg



def fft(img):
    f = np.fft.fft2(img,axes=(0,1))
    return f 


def cor_fft(x1,x2,sigma):
    dist11 = np.sum(np.square(x1))
    dist22 = np.sum(np.square(x2))
    if len(x1.shape)==2:
        c = np.fft.ifft2((np.conj(fft(x1))*fft(x2)))
    else:
        c = np.fft.ifft2(np.sum(np.conj(fft(x1))*fft(x2),2))
    dist= dist11-2*c+dist22
    cor = np.exp(-1*dist/(sigma**2*x1.size))
    cor = np.real(cor)
    return cor

def train(x,y,sigma,l):
    k = cor_fft(x,x,sigma)
    alpha = fft(y)/(fft(k)+l)
    return alpha

def detect(alpha,x,z,sigma):
    k = cor_fft(z,x,sigma)
    response = np.real(np.fft.ifft2(alpha*fft(k)))
    return response

def update_tracker(response,img_size,pos,HOG_flag,scale_factor=1):
    start_w,start_h = response.shape
    w,h = img_size
    px,py,ww,wh = pos
    res_pos = np.unravel_index(response.argmax(),response.shape)
    scale_w = 1.0*scale_factor*(ww*2)/start_w
    scale_h = 1.0*scale_factor*(wh*2)/start_h
    move = list(res_pos)
    if not HOG_flag:
        px_new = [px+1.0*move[0]*scale_w,px-(start_w-1.0*move[0])*scale_w][move[0]>start_w/2] 
        py_new = [py+1.0*move[1]*scale_h,py-(start_h-1.0*move[1])*scale_h][move[1]>start_h/2]
        px_new = np.int(px_new) 
        py_new = np.int(py_new)
    else:
        move[0] = np.floor(res_pos[0]/32.0*(2*ww))
        move[1] = np.floor(res_pos[1]/32.0*(2*wh))
        px_new = [px+move[0],px-(2*ww-move[0])][move[0]>ww] 
        py_new = [py+move[1],py-(2*wh-move[1])][move[1]>wh] 
    if px_new<0: px_new = 0
    if px_new>w: px_new = w-1
    if py_new<0: py_new = 0
    if py_new>h: py_new = h-1
    ww_new = np.ceil(ww*scale_factor)
    wh_new = np.ceil(wh*scale_factor)
    new_pos = (px_new,py_new,ww_new,wh_new)
    return new_pos

def get_window(img, bbox, padding, scale_factor=1 ,rez_shape=None):
    (x,y,w,h) = bbox
    ix,iy = img.shape[0],img.shape[1]
    center_x = np.int(x+np.floor(w/2.0))
    center_y = np.int(y+np.floor(h/2.0))
    w = np.floor(1.0*w*scale_factor)
    h = np.floor(1.0*h*scale_factor)
    x_min,x_max = center_x-np.int(w*padding/2.0),center_x+np.int(w*padding/2.0)
    y_min,y_max = center_y-np.int(h*padding/2.0),center_y+np.int(h*padding/2.0)
    if (x_max-x_min)%2!=0:
        x_max+=1
    if (y_max-y_min)%2!=0:
        y_max+=1
    lx = 0 if x_min>=0 else  x_min*-1
    ly = 0 if y_min>=0 else  y_min*-1
    rx = 0 if x_max<=ix else x_max-ix
    ry = 0 if y_max<=iy else y_max-iy
    x_min = x_min if lx==0 else 0  
    y_min = y_min if ly==0 else 0 
    x_max = x_max if rx==0 else ix
    y_max = y_max if ry==0 else iy
    ww,hh = x_max-x_min,y_max-y_min
    window = (x_min,y_min,ww,hh)
    img_crop = crop_img(img,window)
    if lx==0 and rx==0 and ly==0 and ry==0:
        if rez_shape is not None:
            return cv2.resize(img_crop,rez_shape[::-1])
        else:
            return img_crop
    else:
        if len(img_crop.shape)==3:
            img_crop = np.pad(img_crop,((lx,rx),(ly,ry),(0,0)),'edge')
        else:
            img_crop = np.pad(img_crop,((lx,rx),(ly,ry)),'edge')
        if rez_shape is not None:
            return cv2.resize(img_crop,rez_shape[::-1])
        else:
            return img_crop

def process_cos(img,cos_window):
    if len(img.shape)==3:
        channel = img.shape[2]
        cos_mc = np.tile(cos_window,(channel,1,1))
        cos_window_out = np.transpose(cos_mc,[1,2,0])
    else:
        cos_window_out = cos_window
    return img*cos_window_out


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  

def getFeature(x,cos_window,HOG_flag=0):
    if HOG_flag:
        x = HOG.hog(x)
    else:
        x = x.astype('float64')
        x = prewhiten(x)
    x = process_cos(x,cos_window)
    return x