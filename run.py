import numpy as np
import cv2 
import tracker
import os
import argparse
import sys
import time
import util

def main(args):
    # Load  arg
    dataset = args.dataset_descriptor
    save_directory = args.save_directory
    img_channel = args.multi_channel
    HOG_flag = args.HOG_feature
    scale_factor = args.scale_factor
    save_flag = args.save_img
    resize = args.resize
    show_result = args.show_result
    padding = 2
    
   

    # Load dataset information and get start position
    title = dataset.split('/')
    title = [ t for t in title if t][-1]
    img_lst = util.load_imglst(dataset)
    bbox_lst = util.load_bbox(os.path.join(dataset+'/groundtruth.txt'),resize)
    py,px,h,w = bbox_lst[0]
    pos = (px,py,w,h)
    o_w,o_h = w,h
    frames = len(img_lst)


    # Get image information and init parameter
    output_sigma_factor = 1 / float(16)
    img = cv2.imread(img_lst[0],img_channel)
    if resize:
        img_size = np.int(img.shape[0]/2),np.int(img.shape[1]/2)
    else:
        img_size = img.shape[:2]
    if HOG_flag:
        target_size = 32,32
        l = 0.0001
        sigma = 0.6
        inter_factor = 0.012
        scale_weight = 0.95
    else:
        target_size = np.int(padding/2*w)*2,np.int(padding/2*h)*2
        l = 0.0001
        sigma = 0.2
        inter_factor = 0.02
        scale_weight = 0.95
    f = inter_factor

    # Generate y label
    output_sigma = np.sqrt(np.prod(target_size)) * output_sigma_factor
    cos_window = np.outer(np.hanning(target_size[0]),np.hanning(target_size[1]))
    y = tracker.generate_gaussian(target_size,output_sigma)
    rez_shape = y.shape

    # Create file to save result
    tracker_bb =[]
    result_file = os.path.join(save_directory,title+'_'+'result.txt')
    file = open(result_file,'w')
    start_time = time.time()

    # Tracking
    for i in range(frames):
        img = cv2.imread(img_lst[i],img_channel)
        if resize:
           img = cv2.resize(img,img_size[::-1])
        if i==0:
            x =  tracker.get_window(img, pos, padding, scale_factor, rez_shape)
            x = tracker.getFeature(x,cos_window,HOG_flag)
            alpha = tracker.train(x,y,sigma,l)
            z = x
            best_scale = 1
        else:
            x = tracker.get_window(img, pos, padding, scale_factor,rez_shape)
            x = tracker.getFeature(x,cos_window,HOG_flag)
            response = tracker.detect(alpha,x,z,sigma)
            best_scale = 1
            peak_res = response.max()
            if scale_factor!=1:
                Allscale = [1.0/scale_factor,scale_factor]
                for scale in Allscale:
                    x = tracker.get_window(img, pos, padding, scale,rez_shape)
                    x = tracker.getFeature(x,cos_window,HOG_flag)
                    res = tracker.detect(alpha,x,z,sigma)
                    if res.max()*scale_weight > peak_res:
                        peak_res = res.max()
                        best_scale = scale
                        response = res



            # Update position x z alpha
            new_pos = tracker.update_tracker(response,img_size,pos,HOG_flag,best_scale)
            x = tracker.get_window(img, new_pos, padding, 1, rez_shape)
            x = tracker.getFeature(x,cos_window,HOG_flag)
            new_alpha = tracker.train(x,y,sigma,l)
            alpha = f*new_alpha+(1-f)*alpha
            new_z = x
            z = (1-f)*z+f*new_z
            pos = new_pos

        # Write the position
        if resize:
            out_pos = [int(pos[1]*2),int(pos[0]*2),int(pos[3]*2),int(pos[2]*2)]
        else:
            out_pos = [pos[1],pos[0],pos[3],pos[2]]
        win_string = [ str(p) for p in out_pos]
        win_string = ",".join(win_string)
        tracker_bb.append(win_string)
        file.write(win_string+'\n')

    duration = time.time()-start_time
    fps = int(frames/duration)
    print ('each frame costs %3f second, fps is %d'%(duration/frames,fps))
    file.close()
    
    result = util.load_bbox(result_file,0)
    if show_result:
        util.display_tracker(img_lst,result,save_flag)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_descriptor', type=str, 
        help='The directory of video and groundturth file')
    parser.add_argument('save_directory', type=str, 
        help='The directory of result file')
    parser.add_argument('--show_result', type=int, 
        help='Show result or not',default=1)
    parser.add_argument('--resize', type=float, 
        help='Resize img or not',default=0)
    parser.add_argument('--multi_channel', type=int, 
        help='Use multi channel image or not',default=1)
    parser.add_argument('--HOG_feature', type=int, 
        help='Use HOG or not',default=0)
    parser.add_argument('--scale_factor', type=float, 
        help='bbox scale factor',default=1)
    parser.add_argument('--save_img', type=int, 
        help='save img or not',default=0)

    return parser.parse_args(argv)
    

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))

