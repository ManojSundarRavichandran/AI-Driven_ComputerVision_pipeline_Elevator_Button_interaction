from pyniryo import *
import time
from camera_code import take_photos

from button_detection import ocr_on_bounding_boxes



ned = NiryoRobot("10.10.10.10")

initial_pos=ned.get_pose()

print(ned.need_calibration())
ned.calibrate_auto()
ned.move_to_home_pose()
time.sleep(3)


#*******************Starting position*************************

initial_pos=(0.1353,-0.0089, 0.2372,-0.056,0.080, -0.063)

ned.move_pose(initial_pos)#The arm goes to the initial position 


time.sleep(2)#The arm waits 2 seconds



#*******************take the picture*************************


take_photos()# The robot takes the photo



#*******************Generate the coordinates of the button*************************

image_path = '/media/asmany/Drive_D/Intelligent_Robotics/cameraphoto.jpg'  # Replace with the path to your image


print('Which floor do you want to go?')
floor_number = input()
time.sleep(1)
# Perform OCR on the bounding boxes in the image
coordinates = ocr_on_bounding_boxes(image_path, floor_number)


if coordinates == None:
    print('Button is already pressed')
    ned.go_to_sleep()

else:
    desired_x = ((coordinates[0]+coordinates[2])/2)/3779.53

    desired_x = float(round(desired_x,3))

    desired_y = ((coordinates[1]+coordinates[3])/2)/3779.53

    desired_y = float(round(desired_y,3))
    print(desired_x, desired_y)



#*******************Inverse kinematics*************************

    x = 0.3067
    y = 0.1711 
    z = 0.4469
    roll = -0.225
    pitch = -0.232
    yaw = 0.499

    desired_pose = (0.3, y -desired_x*2.6, z-desired_y*2, 0, 0, 0)  # X, Y, Z, roll, pitch, yaw

    joint_angles = ned.inverse_kinematics(desired_pose)

    ned.move_joints(joint_angles)


#*******************press the button*************************

    press=(0.3+0.02, y -desired_x*2.6, z-desired_y*2, 0, 0, 0)
    joint_angles=ned.inverse_kinematics(press)
    ned.move_joints(joint_angles)

    time.sleep(1)

    press=(0.3-0.02, y -desired_x*2.6, z-desired_y*2, 0, 0, 0)
    joint_angles=ned.inverse_kinematics(desired_pose)
    ned.move_joints(joint_angles)
    time.sleep(5)


    ned.go_to_sleep()



# def inverse_kinematics(x, y, z):


#     return theta1, theta2, theta3, theta4, theta5, theta6


# button_x = 0.2  
# button_y = 0.2
# button_z = 0.2

# theta1, theta2, theta3, theta4, theta5, theta6 = inverse_kinematics(button_x, button_y, button_z)
