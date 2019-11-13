import setup_path 
import airsim
import numpy as np
import sys
import time
import datetime
import threading
import os 
# connect to the AirSim simulator
class straight_crash:
	def __init__(self):	
		self.client = airsim.MultirotorClient()
		self.client.confirmConnection()
		self.client.enableApiControl(True)
		self.client.armDisarm(True)
		self.client.takeoffAsync().join()
		self.z_height = -10 #Define height to run trials at  
		self.speed = 5 #Define speed to fly straight at
		self.alpha = 5000 #Scale's how far ahead next waypoint is, where x_new = x - alpha*sin(theta), y_new = y + alpha*cos(theta)
		self.x_min = -10
		self.x_max = -5
		self.y_min = -5
		self.y_max = 5
		self.pitch = 0
		self.roll = 0
		self.flight_num = 0
		self.file_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'straight_path_images/flight'))
		self.time_step = .15 #Time between images taken 
		self.im_list = []
	#Define initial position and heading and set pose to match these, should call this after every colision
	def reset_pose(self):
		#x = np.random.uniform(self.x_min, self.x_max)
		#y = np.random.uniform(self.x_min, self.x_max)
		#yaw = np.random.uniform(-np.pi, np.pi)
		print ("resetting pose")
		x = np.random.uniform(self.x_min, self.x_max)
		y = np.random.uniform(self.y_min, self.y_max)
		print("New pose")
		yaw = np.random.uniform(-np.pi, np.pi)
		position = airsim.Vector3r(x , y, self.z_height)
		heading = airsim.utils.to_quaternion(self.pitch, self.roll, yaw)
		pose = airsim.Pose(position, heading)
		print("set_yaw")
		self.client.simSetVehiclePose(pose, True) #Set yaw angle
		self.client.moveToPositionAsync(x, y, self.z_height, 5).join() 
		self.client.hoverAsync().join()
		time.sleep(1)
		#Generate waypoint very far in direction drone is facing
		self.waypoint(x, y, yaw)

	def waypoint(self, x, y, yaw):
		#Generate waypoint very far in direction drone is facing
		print ("Moving vehicles")
		x = x + self.alpha * np.cos(yaw)
		y = y + self.alpha * np.sin(yaw)
		print("Move vehicle to ")
		print(x, y, self.z_height)
		self.client.moveToPositionAsync(x, y, self.z_height, 5)
		self.im_list = [] #Store list of images of flight
		while True: #Keep moving to position and storing images until a crash happens
			self.im_store()
			time.sleep(self.time_step) #More or less store images every timestep
	 		if self.client.simGetCollisionInfo().has_collided == True:
	 			#self.im_thread.cancel()#Kill thread 
	 			print ("collision")
	 			self.client.reset()
	 			self.client.enableApiControl(True)
				self.client.armDisarm(True)
				self.image_handle() #Save relevant images to a file
				self.flight_num += 1 #Increase fligh number for saving images
				self.reset_pose()

	def im_store(self):
		#Store images on timer from thread
		png_image = self.client.simGetImages([airsim.ImageRequest("front_center", airsim.ImageType.Scene)]) #RGB png image
		self.im_list.append(png_image[0])


	def image_handle(self):
		#Saves images stored in im_list from flight 
		if len(self.im_list) >= 10:
			for idx, im in enumerate(self.im_list[0:5]):
				airsim.write_file(os.path.normpath(self.file_path + "_" + "safe"  + '_' +  str(self.flight_num)  + '_' + str(idx)  + '.png'), im.image_data_uint8)
			for idx, im in enumerate(self.im_list[len(self.im_list) - 5:]):
				airsim.write_file(os.path.normpath(self.file_path + "_"  + "danger" + str(self.flight_num)  + '_' + str(idx)  + '.png'), im.image_data_uint8)



if __name__ == '__main__':
	x = straight_crash()
	x.reset_pose()
 