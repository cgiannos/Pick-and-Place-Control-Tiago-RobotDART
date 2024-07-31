import numpy as np
import RobotDART as rd
import py_trees
import dartpy
import copy
from utils import create_grid, box_into_basket

dt = 0.001
simulation_time = 100000000.0
total_steps = int(simulation_time / dt)



# Create robot
packages = [("tiago_description", "tiago/tiago_description")]
robot = rd.Tiago(int(1. / dt), "tiago/tiago_steel.urdf", packages)

arm_dofs = ["arm_1_joint", "arm_2_joint", "arm_3_joint", "arm_4_joint", "arm_5_joint", "arm_6_joint", "arm_7_joint", "gripper_finger_joint", "gripper_right_finger_joint"]
robot.set_positions(np.array([np.pi/2., np.pi/4., 0., np.pi/2., 0. , 0., np.pi/2., 0.03, 0.03]), arm_dofs)

# Control base - we make the base fully controllable
robot.set_actuator_type("servo", "rootJoint", False, True, False)
#robot.set_commands([0.1, 0.1, 0.], ['rootJoint_rot_z', 'rootJoint_pos_x', 'rootJoint_pos_y'])

# Create position grid for the box/basket
basket_positions, box_positions = create_grid()

# Create box
box_size = [0.04, 0.04, 0.04]
# Random cube position
box_pt = np.random.choice(len(box_positions))
box_pose = [0., 0., 0., box_positions[box_pt][0], box_positions[box_pt][1], box_size[2] / 2.0]
box = rd.Robot.create_box(box_size, box_pose, "free", 0.1, [0.9, 0.1, 0.1, 1.0], "box_" + str(0))

# Create basket
basket_packages = [("basket", "/home/giannos/tiago_pick_place/models/basket")]
basket = rd.Robot("/home/giannos/tiago_pick_place/models/basket/basket.urdf", basket_packages, "basket")
# Random basket position
basket_pt = np.random.choice(len(basket_positions))
basket_z_angle = 0.
basket_pose = [0., 0., basket_z_angle, basket_positions[basket_pt][0], basket_positions[basket_pt][1], 0.0008]
basket.set_positions(basket_pose)
basket.fix_to_world()

# Create Graphics
gconfig = rd.gui.Graphics.default_configuration()
gconfig.width = 1280
gconfig.height = 960
graphics = rd.gui.Graphics(gconfig)

# Create simulator object
simu = rd.RobotDARTSimu(dt)
simu.set_collision_detector("bullet")
simu.set_control_freq(100)
simu.set_graphics(graphics)
graphics.look_at((0., 4.5, 2.5), (0., 0., 0.25))
simu.add_checkerboard_floor()
simu.add_robot(robot)
simu.add_robot(box)
simu.add_robot(basket)

#controller
class PITask:
    def __init__(self, target, dt, Kp = 10., Ki = 0.1):
        self._target = target
        self._dt = dt
        self._Kp = Kp
        self._Ki = Ki
        self._sum_error = 0
    
    def set_target(self, target):
        self._target = target
    
    # function to compute error
    def error(self, tf):
        # 2 ways of computing rotation error in world frame
        # # 1st way: compute error in body frame
        # error_in_body_frame = rd.math.logMap(tf.rotation().T @ self._target.rotation())
        # # transform it in world frame
        # error_in_world_frame = error_in_body_frame @ tf.rotation().T
        # 2nd way: compute error directly in world frame
        rot_error = rd.math.logMap(self._target.rotation() @ tf.rotation().T)
        lin_error = self._target.translation() - tf.translation()
        return np.r_[rot_error, lin_error]
    
    def update(self, current):
        error_in_world_frame = self.error(current)

        self._sum_error = self._sum_error + error_in_world_frame * self._dt

        return self._Kp * error_in_world_frame + self._Ki * self._sum_error
#sequence behavior 
class ReachTarget(py_trees.behaviour.Behaviour):
    def __init__(self, robot, tf_desired, dt, link, sequence, name="ReachTarget"):
        super(ReachTarget, self).__init__(name)
        # robot
        self.robot = robot
        # end-effector name
        self.eef_link_name = link
        #use A,B,C,D,E,F, for every different movement sequence
        self.sequence=sequence
        # set target tf
        self.tf_desired = tf_desired
        # dt
        self.dt = dt

        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def setup(self):
        self.logger.debug("%s.setup()->does nothing" % (self.__class__.__name__))

#initialize controller and change Kp, Ki parameters of controller so we can have movements with different speeds for each task
    def initialise(self):
        self.logger.debug("%s.initialise()->init controller" % (self.__class__.__name__))
        #controller parameters for robot movement going to basket
        if self.sequence in ['E']:
            self.Kp = 0.1
            self.Ki = 0.01
        #controller parameters for robot movement going to box
        elif self.sequence in ['A']:
            self.Kp = 1
            self.Ki = 0.1
        #controller parameters for every other movement
        else:
            self.Kp = 4.
            self.Ki = 0.1
        #initialize controller
        self.controller = PITask(self.tf_desired, self.dt, self.Kp, self.Ki)

     #runs every sequence of the behavior tree
    def update(self):
        new_status = py_trees.common.Status.RUNNING
        # control the robot
        tf = self.robot.body_pose(self.eef_link_name)
        vel = self.controller.update(tf)
        jac = self.robot.jacobian(self.eef_link_name) # this is in world frame
        jac_pinv = np.linalg.pinv(jac) # np.linalg.pinv(jac) # get pseudo-inverse
        cmd = jac_pinv @ vel

        #these cases make up the different commands that we give to the robot
        #for every movement certain robot commands are unusable(0) and gripper commands are handmade

        #robot goes close to box
        if self.sequence in ['A']:
            #eliminate any sideways or vertical body movement
            cmd[0:2]=0
            #eliminate any arm movement
            cmd[5:]=0
            #keep gripper open
            cmd[24]=0.01

        #arm goes close to box and gripper goes in position  to grab
        elif self.sequence in ['B']:
            #eliminate any body movement except of horizontal rotation
            cmd[0:2]=0
            cmd[3:6]=0
            #keep gripper open
            cmd[24]=0.01

        #short movement just for the gripper to grab box
        elif self.sequence in ['C']:
            #eliminate any body movement
            cmd[0:6]=0
        
        #arm resets its position while holding box
        elif self.sequence in ['D']:
            #eliminate any body movement
            cmd[0:6]=0
            cmd[16]=0
            #keep gripper close
            cmd[24]=-0.01

        #body moves close to basket
        elif self.sequence in ['E']:
            #eliminate any sideways or vertical body movement
            cmd[0:2]=0
            #eliminate any arm movement
            cmd[5:]=0
            #keep gripper close
            cmd[24]=-0.01

        # arm goes in the basket to make sure the box goes in the basket
        elif self.sequence in ['F']:
            #eliminate any body movement
            cmd[0:6]=0
            cmd[16]=0
            #keep gripper close
            cmd[24]=-0.01

        #arm drops box into basket and resets its position
        elif self.sequence in ['G']:
            #eliminate any body movement
            cmd[0:6]=0
            cmd[16]=0
            #open gripper
            cmd[24]=0.01
        #print(cmd)
        self.robot.set_commands(cmd)

        # if error too small, report success
        #for every different movement different error is used for more slow and accurate movement or for faster and unaccurate movement
        err = np.linalg.norm(self.controller.error(tf))
        #error for arm moving to box
        if self.sequence in ['B']:
            if err < 0.057:
                new_status = py_trees.common.Status.SUCCESS
        #error for body moving to box and body moving to basket
        if self.sequence in ['A','E']:
            if err < 0.13:
                new_status = py_trees.common.Status.SUCCESS
        #error for gripper grabin box
        if self.sequence in ['C']:
            if err < 0.09:
                new_status = py_trees.common.Status.SUCCESS
        #error for arm reseting after picking box and for arm going into basket
        if self.sequence in ['F','D']:
            if err < 0.08:
                new_status = py_trees.common.Status.SUCCESS
        #error for reseting arm after dropping box in basket 
        if self.sequence in ['G']:
            if err < 0.08:
                #after the last movement a new random robot position is calculated so that the robot goes after a new box position
                box_pt = np.random.choice(len(box_positions))
                box_pose = [0., 0., 0., box_positions[box_pt][0], box_positions[box_pt][1], box_size[2] / 2.0]
                box.set_base_pose(box_pose)
                #change_base_pos is called to initialize all movements in behavior tree with new box position
                change_base_pos(box.base_pose(),root)
                new_status = py_trees.common.Status.SUCCESS
        if new_status == py_trees.common.Status.SUCCESS:
            self.feedback_message = "Reached target"
            self.logger.debug("%s.update()[%s->%s][%s]" % (self.__class__.__name__, self.status, new_status, self.feedback_message))
        else:
            self.feedback_message = "Error: {0}".format(err)
            self.logger.debug("%s.update()[%s][%s]" % (self.__class__.__name__, self.status, self.feedback_message))
        return new_status

    def terminate(self, new_status):
        self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

#function that initializes the whole behavior tree and its target positions with parameter box position so that we have 50 different controllers
def change_base_pos(base_pose,root):
    #remove children of root every time behavior tree is initialized
    root.remove_all_children()  
    # Create sequence node (for sequential targets)
    sequence = py_trees.composites.Sequence(name="Sequence")

    # Target A , target is close to box ,move robot body close to box
    eef_link_name = "base_link"
    tf_desired = base_pose

    tf = dartpy.math.Isometry3()
    tf.set_translation([tf_desired.translation()[0]-0.52, tf_desired.translation()[1], tf_desired.translation()[2]])

    tf.set_rotation(tf_desired.rotation())
    trA = ReachTarget(robot, tf, dt, eef_link_name, 'A',"Reach Target A")
    # add target to sequence node
    sequence.add_child(trA)

    # Target B , target is closer to box, move arm close to box
    eef_link_name = "gripper_link"
    tf_desired=base_pose

    tf = dartpy.math.Isometry3()
    tf.set_translation([tf_desired.translation()[0]+0.03, tf_desired.translation()[1], tf_desired.translation()[2]+0.1])
    tf.set_rotation(tf_desired.rotation())

    trB = ReachTarget(robot, tf, dt, eef_link_name, 'B', "Reach Target B")
    # add target to sequence node
    sequence.add_child(trB)

    # Target C  , target is closer to box, gripper grabs box
    eef_link_name = "gripper_left_finger_link"
    tf_desired=base_pose

    tf = dartpy.math.Isometry3()
    tf.set_translation([tf_desired.translation()[0]-0.1, tf_desired.translation()[1], tf_desired.translation()[2]+0.1])
    tf.set_rotation(tf_desired.rotation())

    trC = ReachTarget(robot, tf, dt, eef_link_name, 'C', "Reach Target C")
    # add target to sequence node
    sequence.add_child(trC)

    # Target D  , target is higher from box, arm resets position while holding box
    eef_link_name = "gripper_link"
    tf_desired=base_pose

    tf = dartpy.math.Isometry3()
    tf.set_translation([tf_desired.translation()[0], tf_desired.translation()[1], tf_desired.translation()[2]+0.5])
    tf.set_rotation(tf_desired.rotation())

    trD = ReachTarget(robot, tf, dt, eef_link_name, 'D', "Reach Target D")
    # add target to sequence node
    sequence.add_child(trD)


    # Target E , target is basket ,body moves close to basket while holding box
    eef_link_name = "base_link"
    tf_desired=basket.base_pose()

    tf = dartpy.math.Isometry3()
    tf.set_translation([tf_desired.translation()[0]-0.52, tf_desired.translation()[1], tf_desired.translation()[2]])

    tf.set_rotation(tf_desired.rotation())

    trE = ReachTarget(robot, tf, dt,  eef_link_name, 'E',"Reach Target E")
    # add target to sequence node
    sequence.add_child(trE)

    # Target F  , target is closer to basket ,arm moves into basket while honding box
    eef_link_name = "gripper_link"
    tf_desired=basket.base_pose()

    tf = dartpy.math.Isometry3()
    tf.set_translation([tf_desired.translation()[0], tf_desired.translation()[1], tf_desired.translation()[2]+0.3])

    tf.set_rotation(tf_desired.rotation())

    trF = ReachTarget(robot, tf, dt,  eef_link_name, 'F',"Reach Target F")
    # add target to sequence node
    sequence.add_child(trF)

    # Target G ,target is higher to basket ,  arm drops box and resets position
    eef_link_name = "gripper_link"
    tf_desired=basket.base_pose()

    tf = dartpy.math.Isometry3()
    tf.set_translation([tf_desired.translation()[0], tf_desired.translation()[1], tf_desired.translation()[2]+0.5])

    tf.set_rotation(tf_desired.rotation())

    trG = ReachTarget(robot, tf, dt,  eef_link_name, 'G',"Reach Target G")
    # add target to sequence node
    sequence.add_child(trG)

    # Add sequence to tree
    root.add_child(sequence)



# Behavior Tree
py_trees.logging.level = py_trees.logging.Level.DEBUG

# Create tree root
root = py_trees.composites.Parallel(name="Root", policy=py_trees.common.ParallelPolicy.SuccessOnOne())

#initialize behavior tree 
change_base_pos(box.base_pose(),root)



finish_counter = 0

# tick once
root.tick_once()


for step in range(total_steps):
    if (simu.schedule(simu.control_freq())):
        # box basket positions for check
        box_translation = box.base_pose().translation()
        basket_translation = basket.base_pose().translation()

        #count every time box goes into basket
        if box_into_basket(box_translation, basket_translation, basket_z_angle):
            finish_counter += 1
            #after counting box into basket the box need to change position so it wont get counted again in the same movement sequence.
            #The final new random position is given when the arm resets its position in G sequence
            box_pt = np.random.choice(len(box_positions))
            box_pose = [0., 0., 0., box_positions[box_pt][0], box_positions[box_pt][1], box_size[2] / 2.0]
            box.set_base_pose(box_pose)

        # 50 times box goes into basket programm stops  
        if (finish_counter >= 50):
            break

    if (simu.step_world()):
        break
    #runs every sequence of the tree
    root.tick_once()
 
