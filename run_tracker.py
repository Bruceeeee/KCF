import numpy as np
import cv2 
import tracker
import os
import argparse
import sys
import time
import kcf_tracker
import util

def main(args):
    # Load  arg
    dataset = args.dataset_descriptor
    save_directory = args.save_directory
    resize = args.resize
    show_result = args.show_result
    padding = 2
    dataformat = 1
    
   

    # Load dataset information and get start position
    title = dataset.split('/')
    title = [ t for t in title if t][-1]
    img_lst = util.load_imglst(dataset)
    bbox_lst = util.load_bbox(os.path.join(dataset+'/groundtruth.txt'),0,dataformat=1)
    print bbox_lst[:2,:]
    py1, px1, py2, px2 = bbox_lst[0]
    pos = (px1, py1, px2, py2)
    frames = len(img_lst)


    
    tracker_bb =[]
    result_file = os.path.join(save_directory,title+'_'+'result.txt')
    file = open(result_file,'w')
    start_time = time.time()

    # Tracking
    for i in range(frames):
        img = cv2.imread(img_lst[i])
        if i==0:
    # Initialize trakcer, img 3 channel, pos(x1,y1,x2,y2)
            kcftracker = kcf_tracker.Kcftracker(img,pos) 
        else:
    # Update position and traking
            pos = kcftracker.updateTraker(img)

        # Write the position
        out_pos = [pos[1],pos[0],pos[3]-pos[1],pos[2]-pos[0]]
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
        util.display_tracker(img_lst,result,save_flag=0)

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

   

    return parser.parse_args(argv)
    

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))

