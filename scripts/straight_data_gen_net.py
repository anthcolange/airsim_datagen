import setup_path 
import airsim
import numpy as np
import sys
import sys
import time
import datetime
from PIL import Image
from torchvision import transforms
import threading
import os 
import cv2
import csv
from random import randrange
sys.path.append("..")
from nets.dronet import DroNetExt
import torch

# connect to the AirSim simulator
class straight_crash:
	def __init__(self, model):	
		self.model = model
		self.client = airsim.MultirotorClient()
		self.client.confirmConnection()
		self.client.enableApiControl(True)
		self.client.armDisarm(True)
		self.client.takeoffAsync().join()
		self.z_height = -1.1 #Define height to run trials at  ##-.5 for realistic environment and xx for office space
		self.speed = 5 #Define speed to fly straight at
		self.alpha = 5000 #Scale's how far ahead next waypoint is, where x_new = x - alpha*sin(theta), y_new = y + alpha*cos(theta)
		self.x_min = -7
		self.x_max = 7
		self.y_min = -5
		self.soft = torch.nn.Softmax(dim=1)
		self.y_max = 10
		self.pitch = 0
		self.roll = 0
		self.alpha = 5000 # = 0
		self.flight_num = 0
		self.time_step = .05 #Time between images taken 
		self.num_frames = 5 #How many frames we want of safe and dangerous for each flight
		self.last_collision_stamp = 0 #initialize timestep for collision
		self.cam_im_list = []
		self.state_list = []
		transform_list = []
		transform_list.append(transforms.Grayscale(num_output_channels=1))
		transform_list.append(transforms.Resize(size=(200,200)))
		transform_list.append(transforms.ToTensor())
		self.transform = transforms.Compose(transform_list)

	#Define initial position and heading and set pose to match these, should call this after every colision
	def reset_pose(self):
		#print ("resetting pose")
		x = np.random.uniform(self.x_min, self.x_max)
		y = np.random.uniform(self.y_min, self.y_max)
		#print("New pose")
		#print (x,y)
		yaw = np.random.uniform(-np.pi, np.pi)
		self.speed = np.random.uniform(1, 5)
		position = airsim.Vector3r(x , y, self.z_height)
		heading = airsim.utils.to_quaternion(self.pitch, self.roll, yaw)
		pose = airsim.Pose(position, heading)
		#print("set_yaw")
		self.client.simSetVehiclePose(pose, True) #Set yaw angle
		self.client.moveToPositionAsync(x,y, self.z_height, .5).join()
		self.client.hoverAsync()
		time.sleep(2)
		#Generate waypoint very far in direction drone is facing
		self.waypoint(x, y, yaw)

	def waypoint(self, x, y, yaw):
		#Generate waypoint very far in direction drone is facing
		#print ("Moving vehicles")
		x = x + self.alpha * np.cos(yaw)
		y = y + self.alpha * np.sin(yaw)
		#print("Move vehicle to ")
		#print(x, y, self.z_height, self.speed)
		self.client.moveToPositionAsync(x, y, self.z_height, self.speed)
		self.cam_im_list = [] #Store list of images of flight
		self.state_list = [] #Store list of drone states
		while True: #Keep moving to position and storing images until a crash happens
			self.im_store()
			collision_info = self.client.simGetCollisionInfo() #Log collision info
			new_time_stamp = collision_info.time_stamp
			#if collision_info.has_collided: ##Apparently might need to change depending on if windows or not 
			if (new_time_stamp != self.last_collision_stamp and new_time_stamp != 0) or collision_info.has_collided: #Check if collision has new timestamp to validate new collision happened
	 			#self.im_thread.cancel()#Kill thread 
	 			self.last_collision_stamp = new_time_stamp
	 			print ("collision")
	 			self.client.reset()
	 			self.client.enableApiControl(True)
	 			self.client.armDisarm(True)
	 			self.client.takeoffAsync()
	 			time.sleep(.5)
	 			self.reset_pose()
			else:
	 			time.sleep(self.time_step) #More or less store images every timestep

	def im_store(self):
		#Store images on timer from thread
		cam_im = self.client.simGetImages([airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)])#RGB png image
		self.cam_im_list.append(cam_im)
		#self.state_list.append(state_info)
		#If have all frames needed, run through network
		print(len(self.cam_im_list))
		if len(self.cam_im_list) == self.num_frames:
			print("run net")
			im_list = self.cam_im_list
			self.net_test(im_list)
			self.cam_im_list = []


	def net_test(self, im_list):
		"""
		input list of images to run through network 
		"""
		im_list_grey = self.preprocess(im_list)
		im_list_grey = im_list_grey.view(1,5,200,200)
		#out = self.model(torch.FloatTensor(im_list_grey))
		out = self.model(im_list_grey)
		result = self.soft(out)
		print("Result:" +  str(result))
		if result.data[0][0] > result.data[0][1]:
			print ("safe: " + str(result.data[0][0]))
		else:
			print("danger:" + str(out.data[0][0]))

	def preprocess(self, im_list):
		# image transforms
		im_list_grey = []
		for img_rgb in im_list:
			img_rgb = np.fromstring(img_rgb[0].image_data_uint8, dtype=np.uint8).reshape(img_rgb[0].height, img_rgb[0].width, 3) 
			img_rgb = np.flipud(img_rgb)
			img_rgb = Image.fromarray(img_rgb)
			im_list_grey.append(self.transform(img_rgb))
		im_list_grey = torch.stack(im_list_grey) 
		return im_list_grey

if __name__ == '__main__':
	model = DroNetExt(5)
	x = straight_crash(model)
	x.reset_pose()
