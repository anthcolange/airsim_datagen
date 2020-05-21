import setup_path 
import airsim
import numpy as np
import sys
import time
import datetime
import math
import threading
import os 
import cv2
import csv
import random
from random import randrange
# connect to the AirSim simulator
class straight_crash:
	def __init__(self):	
		self.client = airsim.MultirotorClient()
		self.client.confirmConnection()
		self.z_height = -1.7 #Define height to run trials at  ##-.5 for realistic environment and xx for office space
		self.speed = 3 #Define speed to fly straight at
		self.x_min = -7
		self.x_max = 7
		self.y_min = -5
		self.y_max = 10
		self.pitch = 0
		self.roll = 0
		self.flight_num = 0
		self.flight_list = []
		self.time_step = .05  #Time between images taken 
		self.alpha = self.speed * self.time_step # 
		self.num_frames = 5 #How many frames we want of safe and dangerous for each flight
		self.gap = .5 #Distance in meters from collision to give danger reading
		self.last_collision_stamp = 0 #initialize timestep for collision
		self.cam_im_list = []
		self.state_list = []
		self.gen_data_directory(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'straight_path_images'))) 

	#Define initial position and heading and set pose to match these, should call this after every colision
	def reset_pose(self):
		#print ("resetting pose")
		x = np.random.uniform(self.x_min, self.x_max)
		y = np.random.uniform(self.y_min, self.y_max)
		print("New pose: " + str(x) + ',' + str(y))
		yaw = np.random.uniform(-np.pi, np.pi)
		position = airsim.Vector3r(x , y, self.z_height)
		heading = airsim.utils.to_quaternion(self.pitch, self.roll, yaw)
		pose = airsim.Pose(position, heading)
		self.client.simSetVehiclePose(pose, True) #Set yaw angle

		return self.waypoint(x, y, yaw, heading)

	def waypoint(self, x, y, yaw, heading):
		#Generate waypoints in direction drone is facing and move it 

		self.cam_im_list = [] #Store list of images of flight
		self.state_list = [] #Store list of drone states
		iters = 0
		while True: #Keep moving to position and storing images until a crash happens
		#https://github.com/microsoft/AirSim-NeurIPS2019-Drone-Racing/issues/111: ./AirSimExe.sh -windowed -NoVSync -BENCHMARK
			x = x + self.alpha * np.cos(yaw)
			y = y + self.alpha * np.sin(yaw)
			position = airsim.Vector3r(x , y, self.z_height)
			pose = airsim.Pose(position, heading)
			self.client.simSetVehiclePose(pose, True) #Set yaw angle
			print(iters)
			curr_depth, h, w = self.im_store()

			min_depth = np.min(np.reshape(curr_depth, (h, w))[int(h/3):int(h*(2/3))][:,int(w/3):int(w * (2/3))])

			#if collision_info.has_collided: ##Apparently might need to change depending on if windows or not 
			if min_depth <= self.gap: #Check if collision has new timestamp to validate new collision happened
				print("collision")
				#self.im_thread.cancel()#Kill thread 
				self.image_handle(min_depth/self.speed) #Save relevant images to a file
				print(self.flight_num)
				return None #Prevent stackoverflow, return None and call reset_pose
			else:
				iters += 1
				if iters >= 200:
					return None

	def im_store(self):
		#Store images on timer from thread
		cam_im, state_info = self.client.simGetImages([airsim.ImageRequest("front_center", airsim.ImageType.Scene), airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanner, pixels_as_float = True)]),  self.client.simGetVehiclePose()#RGB png image
		self.cam_im_list.append(cam_im)
		self.state_list.append(state_info)
		return cam_im[1].image_data_float, cam_im[1].height, cam_im[1].width


	def image_handle(self, ttc):
		#Saves images stored in im_list from flight and logs times in nanoseconds
		#Depth image handling: https://github.com/microsoft/AirSim/issues/921
		if len(self.cam_im_list) >= 2*self.num_frames: #Check if enough frames to generate safe and danger images


			#Define range of indexes to sample from for safe images
			min_ind = 0
			max_ind = len(self.cam_im_list) - 2*self.num_frames + 1
			start_ind = self.generate_index(min_ind, max_ind) #Generate start index for 5 safe images


			#Store relevant safe images and data associated
			for idx, im in enumerate(self.cam_im_list[start_ind:start_ind + self.num_frames]):

				#Generate time to collision for current frame (time to collision at last frame, + number_time_steps_from_last_frame * dt)
				time_to_collision = ttc + (len(self.cam_im_list) - start_ind - 1 - idx) * self.time_step 

				#Save rgb image to file 
				airsim.write_file(os.path.normpath(os.path.join(self.fold_path, 'flight' + "_" + "rgb" + "_" +"safe"  + '_' +  str(self.flight_num)  + '_' + str(idx) + '.png')), im[0].image_data_uint8)

				#Generate depth image
				depth = im[1]
				depth_im = self.generate_depth_image(depth)
				
				#Save depth image
				np.save(os.path.normpath(os.path.join(self.fold_path, 'flight' +  "_" + "depth" + "_" +"safe"  + '_' +  str(self.flight_num)  + '_' + str(idx))), depth_im)
				#cv2.imwrite(os.path.normpath(os.path.join(self.fold_path, 'flight' +  "_" + "depth" + "_" +"safe"  + '_' +  str(self.flight_num)  + '_' + str(idx)+  '.png')), depth_im)

				#find index of state being used, relative to initial list so can get state info
				state_ind = start_ind + idx 

				#Store state information and save to csv
				row = [str(self.flight_num), 'safe', str(idx), str(self.speed), str(time_to_collision), str(start_ind), str(len(self.cam_im_list))]
				with open(self.csv_path, 'a', newline='') as csvFile:
					writer = csv.writer(csvFile)
					writer.writerow(row)

			#Store relevant danger images and data associated
			for idx, im in enumerate(self.cam_im_list[len(self.cam_im_list) - self.num_frames:]):

				#Generate time to collision for current frame
				time_to_collision = ttc + (self.num_frames - idx - 1) * self.time_step

				#Save rgb image to file
				airsim.write_file(os.path.normpath(os.path.join(self.fold_path,'flight' +  "_"  + "rgb" + "_" +  "danger" + '_' + str(self.flight_num)  + '_' + str(idx) + '.png')), im[0].image_data_uint8)
				
				#Generate depth image
				depth = im[1]
				depth_im = self.generate_depth_image(depth)

				#Save depth image
				np.save(os.path.normpath(os.path.join(self.fold_path,'flight' +  "_"  + "depth" + "_" +"danger"  + '_' +  str(self.flight_num)  + '_' + str(idx))), depth_im)
				#cv2.imwrite(os.path.normpath(os.path.join(self.fold_path,'flight' "_"  + "depth" + "_" +"danger"  + '_' +  str(self.flight_num)  + '_' + str(idx) +  '.png')), depth_im)

				#find index of state being used, relative to initial list so can get state info
				state_ind = len(self.state_list) - self.num_frames + idx #Want state corresponding to 

				#Store state information
				row = [str(self.flight_num), 'danger', str(idx), str(self.speed), str(time_to_collision), str(start_ind), str(len(self.cam_im_list))]
				with open(self.csv_path, 'a', newline='') as csvFile:
					writer = csv.writer(csvFile)
					writer.writerow(row)

			self.flight_num += 1 #Increase fligh number for saving images

	def gen_data_directory(self, file_path):
		"""
		file_path: Path to where we create new folder
		"""
		folders = [int(name) for name in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, name))] #List of folders currently in data directory
		if len(folders) == 0: #If no folders, create new one starting at 0
			new_folder = '0'
		else:  #Otherwise name is one greater than largest folder number
			new_folder = str(max(folders) + 1)
		os.mkdir(os.path.join(file_path, new_folder)) #Create new folder to store data in 
		self.fold_path = os.path.join(file_path, new_folder)
		self.csv_path = os.path.join(self.fold_path, "data.csv") #Create path for csv file
		self.text_path = os.path.join(self.fold_path, "params.txt") 
		with open(self.csv_path, 'w', newline='') as csvFile: #Create new csv file, need to make new folder for each run first
			writer = csv.writer(csvFile)
			writer.writerow(['flight_num','image_label','label_num', "commanded_speed", "time_to_collision", "safe_frame_start", "total_frames"])
		with open(self.text_path, 'w') as out: #Create text file, need to make new folder for each run first
			line1, line2, line3, line4, line5, line6, line7, line8, line9, line10 = 'z_height: ' + str(self.z_height), "speed: " + str(self.speed), "x_min: " + str(self.x_min), "x_max: " + str(self.x_max), "y_min: " + str(self.y_min), "y_max: " + str(self.y_max), "alpha: " + str(self.alpha), "num_frames: " + str(self.num_frames), "gap: " + str(self.gap), "time step: " + str(self.time_step)
			out.write('{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n'.format(line1, line2, line3, line4, line5, line6, line7, line8, line9, line10))

	def generate_index(self, min_ind, max_ind):
		"""
		INPUT: 
		min_ind: Minimum index to sample safe images from, typically 0
		max_ind: Max index to sample safe images from, in theory 5 back from the beginning of danger sampling
		OUTPUT:
		ind: Index from which to begin taking safe images from
		"""

		ind = max_ind -  math.floor(abs(random.random() - random.random()) * (1 + max_ind - min_ind) + min_ind)
		return ind

	def generate_depth_image(self, depth):
		depth_2d = np.reshape(depth.image_data_float, (depth.height, depth.width))
		return depth_2d


if __name__ == '__main__':
	x = straight_crash()
	while x.flight_num < 1200:
		_ = x.reset_pose()
