# airsim_datagen
Generates drone flight fpv images frames using airsim simulator.

## Path generation
Straight line trajectory data generation occurs in straight_data_gen.py. This picks a random starting point and commands the drone  to fly straight until a collision occurs. After a collision a new starting point is picked and the process repeats.
Each of these trajectories has the fpv frames of the drone saved and the first and last 5 images are ultimately saved.


## Image storage
*/straight_path_images*
1. Stores images from straight line trajectories
2. Naming
  **flight_(type)_(flight_num)_(im_num).png**  
		 -**type = {danger, safe}**  
        -danger means frame is of final 5 taken before crash  
        -safe means frame is of initial 5 as start of flight  
      -flight_num is which flight the image corresponds to, using 0 indexing  
      -im_num is which frame of {danger, safe} the image is, using 0 indexing
      
