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
		self.client.enableApiControl(True)
		self.client.armDisarm(True)
		self.client.takeoffAsync().join()
		self.z_height = -.6 #Define height to run trials at  ##-.5 for realistic environment and xx for office space
		self.speed = 3 #Define speed to fly straight at
		self.alpha = 5000 #Scale's how far ahead next waypoint is, where x_new = x - alpha*sin(theta), y_new = y + alpha*cos(theta)
		self.x_min = -11
		self.x_max = 10
		self.y_min = -7
		self.y_max = 6
		self.pitch = 0
		self.roll = 0
		self.alpha = 5000 # = 0
		self.flight_num = 1140
		self.flight_list = []
		self.wait_frames = 20
		self.time_step = .05  #Time between images taken 
		self.num_frames = 5 #How many frames we want of safe and dangerous for each flight
		self.gap = 0.4 #Distance in meters from collision to give danger reading
		self.last_collision_stamp = 0 #initialize timestep for collision
		self.cam_im_list = []
		self.state_list = []
		self.gen_data_directory(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'straight_path_images'))) 

	#Define initial position and heading and set pose to match these, should call this after every colision
	def reset_pose(self):
		#print ("resetting pose")
		x = np.random.uniform(self.x_min, self.x_max)
		y = np.random.uniform(self.y_min, self.y_max)
		#x = 0
		#y = 0
		#yaw = -np.pi/2
		print("New pose: " + str(x) + ',' + str(y))
		yaw = np.random.uniform(-np.pi, np.pi)
		position = airsim.Vector3r(x , y, self.z_height)
		heading = airsim.utils.to_quaternion(self.pitch, self.roll, yaw)
		pose = airsim.Pose(position, heading)
		#print("set_yaw")
		self.client.simSetVehiclePose(pose, True) #Set yaw angle
		self.client.moveToPositionAsync(x,y, self.z_height, .5).join()
		self.client.hoverAsync()
		time.sleep(2)
		#Generate waypoint very far in direction drone is facing
		return self.waypoint(x, y, yaw)

	def waypoint(self, x, y, yaw):
		#Generate waypoint very far in direction drone is facing
		#print ("Moving vehicles")
		pose = self.client.simGetVehiclePose()
		x = pose.position.x_val
		y = pose.position.y_val
		z = pose.position.z_val
		x = x + self.alpha * np.cos(yaw)
		y = y + self.alpha * np.sin(yaw)
		#print("Move vehicle to: ")
		#print(x, y, self.z_height)
		self.client.moveToPositionAsync(x, y, z, self.speed, drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode = airsim.YawMode(False,yaw))

		#self.client.moveByVelocityAsync(self.speed* np.cos(yaw),self.speed * np.sin(yaw) ,0,10)
		self.cam_im_list = [] #Store list of images of flight
		self.state_list = [] #Store list of drone states
		iters = 0
		while True: #Keep moving to position and storing images until a crash happens
		#https://github.com/microsoft/AirSim-NeurIPS2019-Drone-Racing/issues/111: ./AirSimExe.sh -windowed -NoVSync -BENCHMARK
			self.client.simPause(True)
			print(iters)
			if iters > self.wait_frames:
				self.im_store()
			collision_info = self.client.simGetCollisionInfo() #Log collision info
			new_time_stamp = collision_info.time_stamp
			#if collision_info.has_collided: ##Apparently might need to change depending on if windows or not 
			if (new_time_stamp != self.last_collision_stamp and new_time_stamp != 0) or collision_info.has_collided: #Check if collision has new timestamp to validate new collision happened

	 			self.client.simPause(False)
	 			#self.im_thread.cancel()#Kill thread 
	 			self.image_handle(collision_info) #Save relevant images to a file
	 			self.last_collision_stamp = new_time_stamp
	 			print ("collision")
	 			print(self.flight_num)
	 			self.client.reset()
	 			self.client.enableApiControl(True)
	 			self.client.armDisarm(True)
	 			self.client.takeoffAsync()
	 			time.sleep(.5)
	 			return None #Prevent stackoverflow, return None and call reset_pose
			else:
				self.client.simPause(False)
				time.sleep(self.time_step)
				iters += 1

	def im_store(self):
		#Store images on timer from thread
		cam_im, state_info = self.client.simGetImages([airsim.ImageRequest("front_center", airsim.ImageType.Scene), airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanner, pixels_as_float = True)]),  self.client.getMultirotorState()#RGB png image
		self.cam_im_list.append(cam_im)
		self.state_list.append(state_info)


	def image_handle(self, collision_info):
		#Saves images stored in im_list from flight and logs times in nanoseconds
		#Depth image handling: https://github.com/microsoft/AirSim/issues/921
		gap_time = math.floor(self.gap / self.speed) #Determine time needed from collision to give danger reading
		del_ind = math.floor(gap_time / self.time_step) + self.num_frames #Number of frames to delete (add num_frames because need full set of frames when giving result
		del self.cam_im_list[-del_ind:]
		del self.state_list[-del_ind:]
		if len(self.cam_im_list) >= 2*self.num_frames: #Check if enough frames to generate safe and danger images


			#Define range of indexes to sample from for safe images
			min_ind = 0
			max_ind = len(self.cam_im_list) - 2*self.num_frames + 1
			start_ind = self.generate_index(min_ind, max_ind) #Generate start index for 5 safe images

			#Store collision time
			collision_time = collision_info.time_stamp 

			#Store relevant safe images and data associated
			for idx, im in enumerate(self.cam_im_list[start_ind:start_ind + self.num_frames]):

				#Generate time to collision for current frame
				time_to_collision = collision_time - im[0].time_stamp 

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
				linear_velocity = self.state_list[state_ind].kinematics_estimated.linear_velocity
				angular_velocity = self.state_list[state_ind].kinematics_estimated.angular_velocity
				x_lin_vel, y_lin_vel, z_lin_vel, x_ang_vel, y_ang_vel, z_ang_vel = linear_velocity.x_val, linear_velocity.y_val, linear_velocity.z_val, angular_velocity.x_val, angular_velocity.y_val, angular_velocity.z_val
				row = [str(self.flight_num), 'safe', str(idx),str(x_lin_vel),str(y_lin_vel), str(z_lin_vel), str(x_ang_vel), str(y_ang_vel), str(z_ang_vel), str(np.sqrt(x_lin_vel**2 + y_lin_vel**2 + z_lin_vel**2)), str(self.speed), str(time_to_collision), str(start_ind), str(len(self.cam_im_list))]
				with open(self.csv_path, 'a') as csvFile:
					writer = csv.writer(csvFile)
					writer.writerow(row)

			#Store relevant danger images and data associated
			for idx, im in enumerate(self.cam_im_list[len(self.cam_im_list) - self.num_frames:]):

				#Generate time to collision for current frame
				time_to_collision = collision_time - im[0].time_stamp #time stamp the same for im[0] and im[1]

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
				linear_velocity = self.state_list[state_ind].kinematics_estimated.linear_velocity
				angular_velocity = self.state_list[state_ind].kinematics_estimated.angular_velocity
				x_lin_vel, y_lin_vel, z_lin_vel, x_ang_vel, y_ang_vel, z_ang_vel = linear_velocity.x_val, linear_velocity.y_val, linear_velocity.z_val, angular_velocity.x_val, angular_velocity.y_val, angular_velocity.z_val
				row = [str(self.flight_num), 'danger', str(idx),str(x_lin_vel),str(y_lin_vel), str(z_lin_vel), str(x_ang_vel), str(y_ang_vel), str(z_ang_vel), str(np.sqrt(x_lin_vel**2 + y_lin_vel**2 + z_lin_vel**2)), str(self.speed), str(time_to_collision), str(start_ind), str(len(self.cam_im_list))]
				with open(self.csv_path, 'a') as csvFile:
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
		with open(self.csv_path, 'w') as csvFile: #Create new csv file, need to make new folder for each run first
			writer = csv.writer(csvFile)
			writer.writerow(['flight_num','image_label','label_num', 'x_lin_vel','y_lin_vel','z_lin_vel','x_ang_vel','y_ang_vel','z_ang_vel', "current_speed", "commanded_speed", "time_to_collision", "safe_frame_start", "total_frames"])
		with open(self.text_path, 'w') as out: #Create new csv file, need to make new folder for each run first
			line1, line2, line3, line4, line5, line6, line7, line8, line9, line10 = 'z_height: ' + str(self.z_height), "speed: " + str(self.speed), "x_min: " + str(self.x_min), "x_max: " + str(self.x_max), "y_min: " + str(self.y_min), "y_max: " + str(self.y_max), "alpha: " + str(self.alpha), "num_frames: " + str(self.num_frames), "gap: " + str(self.gap), 'wait_frames:' + str(self.wait_frames)
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

		depth_float = np.array(depth.image_data_float, dtype=np.float32)
		depth_2d = depth_float.reshape(depth.height, depth.width)
		#depth_im = np.array(depth_2d * 255, dtype=np.uint8)
		return depth_2d


if __name__ == '__main__':
	x = straight_crash()
	while x.flight_num < 1200:
		_ = x.reset_pose()
