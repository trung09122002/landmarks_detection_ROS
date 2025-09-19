# landmarks_detection_ROS

A ROS package for real-time landmark detection and recognition using YOLO object detection and Bag-of-Visual-Words (BoVW) approach with Intel RealSense camera.

## 📋 Overview

This ROS package provides a node that combines YOLO11 object detection with Bag-of-Visual-Words (BoVW) for landmark recognition. It processes RGB-D data from Intel RealSense camera and publishes detected landmarks as custom ROS messages for integration with SLAM systems.

## ✨ Features

- **Real-time landmark detection** using YOLO11
- **Bag-of-Visual-Words (BoVW)** for landmark recognition  
- **RGB-D support** with Intel RealSense camera
- **Custom ROS messages** (`landmark` and `landmark_array`)
- **RViz visualization** support with markers
- **Configurable parameters** for detection thresholds
- **SIFT feature extraction** for robust landmark matching

## 🔧 Dependencies

### ROS Packages
```bash
sudo apt-get install ros-noetic-realsense2-camera
sudo apt-get install ros-noetic-cv-bridge
sudo apt-get install ros-noetic-rospkg
sudo apt-get install ros-noetic-visualization-msgs
```

### Python Packages
```bash
pip install ultralytics
pip install opencv-python
pip install scikit-learn
pip install numpy
pip install joblib
```

## 📦 Installation

1. **Clone and build:**
```bash
cd ~/landmarks_detection_ROS/src
# Clone your repository here
cd ~/landmarks_detection_ROS
catkin_make
source devel/setup.bash
```

2. **Install Python dependencies:**
```bash
pip install ultralytics opencv-python scikit-learn joblib
```

## 🚀 Usage

### Launch the landmark detection node:

```bash
# Source your workspace
source ~/landmarks_detection_ROS/devel/setup.bash

# Launch with RealSense camera
roslaunch landmarks_detection landmark.launch
```

### Launch only the detection node (if camera is already running):

```bash
rosrun landmarks_detection landmark.py
```

### Visualize in RViz:

```bash
rosrun rviz rviz -d $(rospack find landmarks_detection)/rviz/landmarks.rviz
```

## ⚙️ Configuration

### Launch File Parameters

Edit `launch/landmark.launch` to customize:

```xml
<param name="yolo_model" value="$(find landmarks_detection)/weights/last.pt"/>
<param name="codebook"   value="$(find landmarks_detection)/codebook/codebook_kNN.joblib"/>
<param name="idf"        value="$(find landmarks_detection)/codebook/idf.npy"/>
<param name="conf_thres" value="0.25"/>      <!-- YOLO confidence threshold -->
<param name="sim_thresh" value="0.12"/>      <!-- BoVW similarity threshold -->
<param name="use_depth"  value="true"/>      <!-- Enable depth data -->
<param name="publish_markers" value="true"/> <!-- Enable RViz markers -->
<param name="show_debug" value="true"/>      <!-- Debug output -->
```

### ROS Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `yolo_model` | string | `weights/last.pt` | Path to YOLO model weights |
| `codebook` | string | `codebook/codebook_kNN.joblib` | Path to trained codebook |
| `idf` | string | `codebook/idf.npy` | Path to IDF weights |
| `conf_thres` | float | 0.5 | YOLO detection confidence threshold |
| `sim_thresh` | float | 0.12 | BoVW similarity threshold |
| `snapshots` | int | 7 | Maximum number of snapshots |
| `use_depth` | bool | true | Enable/disable depth data |
| `use_surf` | bool | false | Use SURF instead of SIFT |
| `depth_topic` | string | `/camera/aligned_depth_to_color/image_raw` | Depth topic |
| `color_topic` | string | `/camera/color/image_raw` | RGB topic |
| `info_topic` | string | `/camera/color/camera_info` | Camera info topic |
| `max_depth_m` | float | 6.0 | Maximum depth in meters |
| `depth_kernel` | int | 7 | Depth processing kernel size |
| `publish_markers` | bool | true | Enable RViz visualization |

## 📡 ROS Interface

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/landmarks` | `landmarks_detection_ROS/landmark_array` | Detected landmarks |
| `/landmark_markers` | `visualization_msgs/MarkerArray` | RViz markers |

### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/color/image_raw` | `sensor_msgs/Image` | RGB camera feed |
| `/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | Depth data |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | Camera parameters |

### Custom Messages

#### Landmark Message (`msg/landmark.msg`)
```yaml
int32 id          # Unique landmark ID
string cls        # Landmark class name
float32 sim       # Similarity score (0-1)
int32[4] bbox     # Bounding box [x1,y1,x2,y2]
float32 x         # 3D position X (meters)
float32 y         # 3D position Y (meters)
float32 z         # 3D position Z (meters)
```

#### Landmark Array Message (`msg/landmark_array.msg`)
```yaml
std_msgs/Header header
landmark[] landmarks
```

## 🔧 Codebook Training

To create a custom codebook for your landmarks:

```bash
cd codebook/
python create_codebook.py
```

This script will:
1. Extract SIFT features from training images in `../dataset/train/images/`
2. Train a MiniBatchKMeans clustering model
3. Compute IDF weights for TF-IDF scoring
4. Save `codebook_kNN.joblib` and `idf.npy`

### Training Dataset Structure

```
dataset/
├── train/
│   ├── images/          # Training images
│   └── labels/          # YOLO labels (optional)
├── valid/
│   ├── images/          # Validation images
│   └── labels/
└── test/
    ├── images/          # Test images
    └── labels/
```

## 🏗️ Package Structure

```
landmarks_detection_ROS/
├── src/
│   └── landmark.py              # Main ROS node
├── launch/
│   └── landmark.launch          # Launch file
├── msg/
│   ├── landmark.msg             # Single landmark message
│   └── landmark_array.msg       # Array of landmarks
├── codebook/
│   ├── codebook_kNN.joblib     # Trained codebook
│   ├── idf.npy                 # IDF weights
│   └── create_codebook.py      # Codebook creation script
├── weights/
│   └── last.pt                 # YOLO model weights
├── dataset/                    # Training dataset
├── rviz/
│   └── landmarks.rviz          # RViz configuration
├── CMakeLists.txt
├── package.xml
└── README.md
```

## 🐛 Troubleshooting

### Common Issues

1. **Package not found:**
```bash
# Make sure to source the workspace
source ~/landmarks_detection_ROS/devel/setup.bash
# Verify package is found
rospack find landmarks_detection_ROS
```

2. **Import errors:**
```bash
# Install missing packages
pip install ultralytics opencv-python scikit-learn joblib
```

3. **Camera not detected:**
```bash
# Test RealSense camera
rosrun realsense2_camera realsense2_camera_manager
# Check topics
rostopic list | grep camera
```

4. **Feature dimension mismatch:**
- Ensure codebook was trained with SIFT features (128 dimensions)
- The node uses SIFT by default, matching the codebook

5. **Permission denied:**
```bash
chmod +x src/landmark.py
```

6. **Message generation errors:**
```bash
# Rebuild the workspace
cd ~/landmarks_detection_ROS
catkin_make clean
catkin_make
source devel/setup.bash
```

### Debug Mode

Enable detailed debug output:
```xml
<param name="show_debug" value="true"/>
```

This will show:
- Detection results
- Feature extraction info
- BoVW matching scores
- 3D position calculations

## 📝 Notes

- **Feature Detector**: Uses SIFT (128 dimensions) by default
- **Camera**: Designed for Intel RealSense D435i
- **Depth Processing**: Uses median filtering for robust depth estimation
- **SLAM Integration**: Compatible with Cartographer ROS
- **Performance**: Optimized for real-time operation

## 🔗 Related Packages

- `realsense2_camera`: Intel RealSense camera driver
- `cv_bridge`: OpenCV-ROS image conversion
- `visualization_msgs`: RViz marker messages

## 📄 License

This package is part of the landmarks detection system. See the main project license for details.

---

**Maintainer:** quangtrung  
**Email:** quangtrung@todo.todo  
**ROS Version:** Noetic
