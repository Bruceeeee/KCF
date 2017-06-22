import tracker
import numpy as np
import cv2

class Kcftracker(object):
	def __init__(self, img, start_pos, HOG_flag=0, dataformat=1, resize=1):
		
		self.HOG_flag = HOG_flag
		self.padding = 2
		self.dataformat = dataformat
		self.resize = resize
		self.img_size = img.shape[0],img.shape[1]

		if self.dataformat:
			w,h = start_pos[2]-start_pos[0],start_pos[3]-start_pos[1]
			self.pos = start_pos[0],start_pos[1],w,h
		else:
			self.pos = start_pos

		if self.resize:
			self.pos = tuple([ele/2 for ele in self.pos])
			self.img_size = img.shape[0]/2, img.shape[1]/2
			img = cv2.resize(img,self.img_size[::-1])

		object_size = self.pos[2:]
		if self.HOG_flag:
			self.target_size = 32,32
			self.l = 0.0001
			self.sigma = 0.6
			self.f = 0.012
		else:
			self.target_size = object_size[0]*self.padding,object_size[1]*self.padding
			self.l = 0.0001
			self.sigma = 0.2
			self.f = 0.02
		output_sigma_factor = 1/float(8)

		output_sigma = np.sqrt(np.prod(self.target_size)) * output_sigma_factor
		self.cos_window = np.outer(np.hanning(self.target_size[0]), np.hanning(self.target_size[1]))
		self.y = tracker.generate_gaussian(self.target_size, output_sigma)
		x =  tracker.get_window(img, self.pos, self.padding)
		x = tracker.getFeature(x, self.cos_window,self.HOG_flag)
		self.alpha = tracker.train(x, self.y, self.sigma, self.l)
		self.z = x
	
     
       

	def updateTraker(self, img):
		 if self.resize:
		 	img = cv2.resize(img,self.img_size[::-1])
		 x = tracker.get_window(img, self.pos, self.padding, 1, self.target_size)
		 x = tracker.getFeature(x, self.cos_window, HOG_flag=0)
		 response = tracker.detect(self.alpha, x, self.z, self.sigma)
		 new_pos = tracker.update_tracker(response, self.img_size, self.pos, HOG_flag=0, scale_factor=1)
		 x = tracker.get_window(img, new_pos, self.padding, 1, self.target_size)
		 x = tracker.getFeature(x, self.cos_window, HOG_flag=0)
		 new_alpha = tracker.train(x, self.y, self.sigma, self.l)
		 self.alpha = self.f*new_alpha + (1-self.f)*self.alpha
		 new_z = x
		 self.z = (1-self.f)*self.z + self.f*new_z
		 self.pos = new_pos
		 output_pos = self.pos
		 print self.pos
		 if self.resize:
		 	output_pos = tuple([ele*2 for ele in self.pos])
		 if self.dataformat:
			output_pos = output_pos[0],output_pos[1],output_pos[0]+output_pos[2],output_pos[1]+output_pos[3]
		 print output_pos
		 return output_pos

