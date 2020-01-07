import setup_path 
import airsim
import numpy as np
import sys
import time
import datetime
import threading
import os 
import cv2
import csv
from random import randrange
# connect to the AirSim simulator
class straight_crash:
	def __init__(self):	
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
		self.y_max = 10
		self.pitch = 0
		self.roll = 0
		self.alpha = 5000 # = 0
		self.flight_num = 0
		self.time_step = .15 #Time between images taken 
		self.num_frames = 5 #How many frames we want of safe and dangerous for each flight
		self.last_collision_stamp = 0 #initialize timestep for collision
		self.cam_im_list = []
		self.state_list = []
		self.gen_data_directory(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'straight_path_images'))) 
	#Define initial position and heading and set pose to match these, should call this after every colision
	def reset_pose(self):
		print ("resetting pose")
		x = np.random.uniform(self.x_min, self.x_max)
		y = np.random.uniform(self.y_min, self.y_max)
		print("New pose")
		print (x,y)
		yaw = np.random.uniform(-np.pi, np.pi)
		self.speed = np.random.uniform(1, 5)
		position = airsim.Vector3r(x , y, self.z_height)
		heading = airsim.utils.to_quaternion(self.pitch, self.roll, yaw)
		pose = airsim.Pose(position, heading)
		print("set_yaw")
		self.client.simSetVehiclePose(pose, True) #Set yaw angle
		self.client.moveToPositionAsync(x,y, self.z_height, .5).join()
		self.client.hoverAsync()
		time.sleep(2)
		#Generate waypoint very far in direction drone is facing
		self.waypoint(x, y, yaw)

	def waypoint(self, x, y, yaw):
		#Generate waypoint very far in direction drone is facing
		print ("Moving vehicles")
		x = x + self.alpha * np.cos(yaw)
		y = y + self.alpha * np.sin(yaw)
		print("Move vehicle to ")
		print(x, y, self.z_height, self.speed)
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
	 			self.image_handle(collision_info) #Save relevant images to a file
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
		cam_im, state_info = self.client.simGetImages([airsim.ImageRequest("front_center", airsim.ImageType.Scene), airsim.ImageRequest("front_center", airsim.ImageType.DepthVis, True)]),  self.client.getMultirotorState()#RGB png image
		self.cam_im_list.append(cam_im)
		self.state_list.append(state_info)


	def image_handle(self, collision_info):
		#Saves images stored in im_list from flight and logs times in nanoseconds
		#Depth image handling: https://github.com/microsoft/AirSim/issues/921
		if len(self.cam_im_list) >= 2*self.num_frames:
			start_ind = randrange(len(self.cam_im_list) - 2*self.num_frames + 1) #Generate random index to begin grabbing frames from 
			collision_time = collision_info.time_stamp
			for idx, im in enumerate(self.cam_im_list[start_ind:start_ind + self.num_frames ]):
				time_to_collision = collision_time - im[0].time_stamp #time stamp the same for im[0] and im[1]
				airsim.write_file(os.path.normpath(os.path.join(self.fold_path, 'flight' + "_" + "rgb" + "_" +"safe"  + '_' +  str(self.flight_num)  + '_' + str(idx) + "_" + str(time_to_collision) + '.png')), im[0].image_data_uint8)
				depth = im[1]
				depth_float = np.array(depth.image_data_float, dtype=np.float32)
				depth_2d = depth_float.reshape(depth.height, depth.width)
				depth_im = np.array(depth_2d * 65535, dtype=np.uint16)
				cv2.imwrite(os.path.normpath(os.path.join(self.fold_path, 'flight' +  "_" + "depth" + "_" +"safe"  + '_' +  str(self.flight_num)  + '_' + str(idx)+  "_" + str(time_to_collision) + '.png')), depth_im)
				state_ind = start_ind + idx #find index of state being used 
				linear_velocity = self.state_list[state_ind].kinematics_estimated.linear_velocity
				angular_velocity = self.state_list[state_ind].kinematics_estimated.angular_velocity
				x_lin_vel, y_lin_vel, z_lin_vel, x_ang_vel, y_ang_vel, z_ang_vel = linear_velocity.x_val, linear_velocity.y_val, linear_velocity.z_val, angular_velocity.x_val, angular_velocity.y_val, angular_velocity.z_val
				#row = [str(self.flight_num), 'safe', str(idx),str(x_lin_vel),str(y_lin_vel), str(z_lin_vel), str(x_ang_vel), str(y_ang_vel), str(z_ang_vel)]
				row = [str(self.flight_num), 'safe', str(idx),str(x_lin_vel),str(y_lin_vel), str(z_lin_vel), str(x_ang_vel), str(y_ang_vel), str(z_ang_vel), self.speed]
				with open(self.csv_path, 'a') as csvFile:
					writer = csv.writer(csvFile)
					writer.writerow(row)
				#print(depth_im)
				# depth1d = np.array(depth.image_data_float, dtype = np.float)
				# depth1d = depth1d*3.5 + 30
				# depth1d[depth1d>255] = 255
				# depth2d = np.reshape(depth1d, (depth.height, depth.width))
				# depth_im = np.array(depth2d, dtype=np.uint8)
				#airsim.write_pfm(os.path.normpath(self.file_path + "_" + "depth" + "_" +"safe"  + '_' +  str(self.flight_num)  + '_' + str(idx)  + '.pfm'), airsim.get_pfm_array(im[1]))
				#airsim.write_file(os.path.normpath(self.file_path + "_" + "depth" + "_" +"safe"  + '_' +  str(self.flight_num)  + '_' + str(idx)  + '.pfm'), depth)
			for idx, im in enumerate(self.cam_im_list[len(self.cam_im_list) - self.num_frames:]):
				time_to_collision = collision_time - im[0].time_stamp #time stamp the same for im[0] and im[1]
				airsim.write_file(os.path.normpath(os.path.join(self.fold_path,'flight' +  "_"  + "rgb" + "_" +  "danger" + '_' + str(self.flight_num)  + '_' + str(idx) + "_" + str(time_to_collision) + '.png')), im[0].image_data_uint8)
				depth = im[1]
				depth_float = np.array(depth.image_data_float, dtype=np.float32)
				depth_2d = depth_float.reshape(depth.height, depth.width)
				depth_im = np.array(depth_2d * 65535, dtype=np.uint16)
				cv2.imwrite(os.path.normpath(os.path.join(self.fold_path,'flight' "_"  + "depth" + "_" +"danger"  + '_' +  str(self.flight_num)  + '_' + str(idx) + "_" +  str(time_to_collision) + '.png')), depth_im)
				state_ind = len(self.state_list) - self.num_frames + idx #Want state corresponding to 
				linear_velocity = self.state_list[state_ind].kinematics_estimated.linear_velocity
				angular_velocity = self.state_list[state_ind].kinematics_estimated.angular_velocity
				x_lin_vel, y_lin_vel, z_lin_vel, x_ang_vel, y_ang_vel, z_ang_vel = linear_velocity.x_val, linear_velocity.y_val, linear_velocity.z_val, angular_velocity.x_val, angular_velocity.y_val, angular_velocity.z_val
				#row = [str(self.flight_num), 'danger', str(idx), str(x_lin_vel),str(y_lin_vel), str(z_lin_vel), str(x_ang_vel), str(y_ang_vel), str(z_ang_vel)]
				row = [str(self.flight_num), 'danger', str(idx),str(x_lin_vel),str(y_lin_vel), str(z_lin_vel), str(x_ang_vel), str(y_ang_vel), str(z_ang_vel), self.speed]
				with open(self.csv_path, 'a') as csvFile:
					writer = csv.writer(csvFile)
					writer.writerow(row)
				# depth1d = np.array(depth.image_data_float, dtype = np.float)
				# depth1d = depth1d*3.5 + 30
				# depth1d[depth1d>255] = 255
				# depth2d = np.reshape(depth1d, (depth.height, depth.width))
				# depth_im = np.array(depth2d, dtype=np.uint8)
				# print (depth_im)
				#depth = np.array(im[1].image_data_float, dtype=np.float32)
				#depth = depth.reshape(im[1].height, im[1].width)
				#depth = np.array(depth * 255, dtype=np.uint8)
				#airsim.write_pfm(os.path.normpath(self.file_path + "_" + "depth" + "_" +"safe"  + '_' +  str(self.flight_num)  + '_' + str(idx)  + '.pfm'), airsim.get_pfm_array(im[1]))
				#airsim.write_file(os.path.normpath(self.file_path + "_"  + "depth" + "_" +  "danger" + str(self.flight_num)  + '_' + str(idx)  + '.pfm'), depth)
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
		with open(self.csv_path, 'w') as csvFile: #Create new csv file, need to make new folder for each run first
			writer = csv.writer(csvFile)
			#writer.writerow(['flight_num','image_label','label_num', 'x_lin_vel','y_lin_vel','z_lin_vel','x_ang_vel','y_ang_vel','z_ang_vel', "time_step:"+str(self.time_step), "speed:"+str(self.speed), "z_height:"+str(self.z_height)])
			writer.writerow(['flight_num','image_label','label_num', 'x_lin_vel','y_lin_vel','z_lin_vel','x_ang_vel','y_ang_vel','z_ang_vel', "speed:"+str(self.speed), "time_step:"+str(self.time_step), "z_height:"+str(self.z_height)])

if __name__ == '__main__':
	x = straight_crash()
	x.reset_pose()
