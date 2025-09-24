cmake_minimum_required(VERSION 3.0.2)
project(landmarks_detection)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  message_generation
  rospy
  sensor_msgs
  std_msgs
)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  landmark.msg
  landmark_array.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES landmarks_detection
  CATKIN_DEPENDS cv_bridge message_generation message_runtime rospy sensor_msgs std_msgs
#  DEPENDS system_lib
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

## Install Python scripts
catkin_install_python(PROGRAMS
  src/landmark.py
  src/landmark_update.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)