import numpy as np
import RobotDART as rd
import dartpy # OSX breaks if this is imported before RobotDART

########## Create simulator object ##########
simu = rd.RobotDARTSimu(0.001)

########## Load our new robot ##########
robot = rd.Robot("four_legs.urdf") # four_legs.urdf should be in the running directory
#robot.fix_to_world()
simu.add_robot(robot)
simu.add_floor()

########## Create Graphics ##########
gconfig = rd.gui.GraphicsConfiguration(1024, 768) # Create a window of 1024x768 resolution/size
graphics = rd.gui.Graphics(gconfig) # create graphics object with configuration
simu.set_graphics(graphics)
graphics.look_at([0., 3., 3.], [0., 0., 0.])

########## Fix robot to world frame ##########
#robot.fix_to_world()

########## Initial positions to allow falling ##########
robot.set_positions([0., 0.])

########## robot_dart Forward Kinematics ##########
print("robot_dart_link_1:")
print(robot.body_pose("fleft_link_1"))
print("robot_dart_link_2:")
print(robot.body_pose("fleft_link_2"))
print("robot_dart_link_3:")
print(robot.body_pose("fleft_link_3"))

########## Run simulation ##########
while True:
    if simu.step_world():
        break