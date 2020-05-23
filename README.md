# airsim_datagen
Generates drone flight fpv images frames using airsim simulator.

## Path generation
Straight line trajectory data generation picks a random starting point and commands the drone to fly straight until a collision occurs. After a collision a new starting point is picked and the process repeats.
Each of these are set to generate 1200 trajectories, where each trajectory saves the last 5 images as a danger label, and a random 5 consecutive frames probabilistically set to sample closer to the end as safe labels. In each case we are saving RGB + Depth images.

All generation scripts in scripts/
1. /straight_data_gen.py
	-Traditional data generation using drone dynamics and collision detection
2. /straight_data_gen_callback(WIP)
	-Same as straight_data_gen but using threaded callback to grab data
3. /straight_data_gen_cvmode
	-Idealized mode with exact dynamics using CV mode. Collision determined by min_depth
4. /straight_data_gen_net
	-Integrates NN to output prediction in real-time (built off of straight_data_gen)


## Image storage
straight_path_images/
1. Each time running a data generation script, a new folder is made here with index one larger than max index of folders in the directory. All data then saved to here.
2. Naming
  **flight_(type)_(label)_(flight_num)_(im_num).png**  
		 -**type = {rgb, depth}**  
		 -**label = {safe, danger}**
        -danger means frame is of final 5 taken before crash  
        -safe means frame is of random 5 before danger images  
      	-flight_num is which flight the image corresponds to, using 0 indexing  
      	-im_num is which frame of {danger, safe} the image is, using 0 indexing
      
## Image settings
settings.json
1. Contains the AirSim settings 
2. Replace files of the same name, stored in ~/Documents/AirSim with this one
3. Images changed to VGA 640x480 
4. CV Mode can be activated here by adding:   "SimMode": "ComputerVision"
5. Headless display can be done with "ViewMode": "NoDisplay"

## TO DO
1. Figure out why NoDisplay mode dims output images
2. Finish or scrap callback method
3. 

## Speed increase (WIP)
Set "ViewMode": "NoDisplay"
Enter ./AirSimExe.sh -windowed -NoVSync -BENCHMARK in Unreal terminal, opened with ~
Changes in clock speed of minimal use when bottlenecked by image grabbing so much

## Notebooks
/Bias_sampling.ipynb
	-Visualize the bias used for sampling safe images along a trajectory
/Ground_Truth_ttc.ipynb
	-Generate the ttcs for a dataset
/KITTI_data_process.ipynb
	-Visualize KITTI ttcs
