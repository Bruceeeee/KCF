import tracker
class Kcftracker(object):
	def __init__(self, img, strat_pos):
		self.img_size = img.shape[:2]
		if self.HOG_flag:
			self.target_size = 32,32
			self.l = 0.0001
			self.sigma = 0.6
			self.f = 0.012
		else:
			self.target_size = img_size[0]*2,img_size[1]*2
			self.l = 0.0001
			self.sigma = 0.2
			self.f = 0.02
		output_sigma_factor = 1/float(8)
		output_sigma = np.sqrt(np.prod(self.target_size)) * output_sigma_factor
		self.cos_window = np.outer(np.hanning(target_size[0]),np.hanning(target_size[1]))
		self.y = tracker.generate_gaussian(self.target_size,output_sigma)
		x =  tracker.get_window(img,start_pos,1)
		x = tracker.getFeature(x,self.cos_window,self.HOG_flag)
		self.alpha = tracker.train(x,self.y,self.sigma,self.l)
		self.z = x
		self.pos = start_pos
     
       

	def updateTraker(self, img):
		 x = tracker.get_window(img,self.pos,1,self.target_size)
		 x = tracker.getFeature(x,self.cos_window,HOG_flag=0)
		 response = tracker.detect(self.alpha,x,self.z,sigma)
		 new_pos = tracker.update_tracker(response,self.img_size,self.pos,HOG_flag=0,best_scale=1)
		 x = tracker.get_window(img,new_pos,1,target_size)
		 x = tracker.getFeature(x,self.cos_window,HOG_flag=0)
		 new_alpha = tracker.train(x,self.y,self.sigma,self.l)
		 self.alpha = self.f*new_alpha + (1-self.f)*self.alpha
		 new_z = x
		 self.z = (1-self.f)*self.z + self.f*new_z
		 self.pos = new_pos
		 return self.pos

