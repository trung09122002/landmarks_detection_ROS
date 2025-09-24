# landmarks_detection

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
- **Advanced re-identification (re-ID)** with multiple algorithms:
  - **ID debouncing** to prevent rapid ID switching
  - **Prototype snapshots** for robust feature matching
  - **Class-specific thresholds** for optimized detection per object type
  - **Spatial scoring** combining appearance, 3D distance, and temporal recency
  - **3D spatial gating** for efficient candidate filtering

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

### Launch the basic landmark detection node:

```bash
# Source your workspace
source ~/landmarks_detection_ROS/devel/setup.bash

# Launch with RealSense camera
roslaunch landmarks_detection landmark.launch
```

### Launch the enhanced landmark detection node with re-ID:

```bash
# Launch with advanced re-identification features
roslaunch landmarks_detection landmark_update.launch
```

### Launch only the detection node (if camera is already running):

```bash
# Basic version
rosrun landmarks_detection landmark.py

# Enhanced version with re-ID
rosrun landmarks_detection landmark_update.py
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

### Enhanced Re-ID Parameters (landmark_update.launch)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_debounce` | bool | true | Enable ID debouncing to prevent ID switching |
| `debounce_window` | int | 5 | Number of frames to keep old ID if match is weak |
| `center_match_pix` | int | 60 | Pixel radius for center-based ID assignment |
| `use_prototype` | bool | true | Enable prototype snapshots for multiple feature vectors per ID |
| `prototype_maxlen` | int | 7 | Maximum number of snapshots per landmark |
| `use_per_class_th` | bool | true | Enable class-specific similarity thresholds |
| `class_thresholds` | dict | - | Per-class similarity thresholds (bottle: 0.10, chair: 0.16, etc.) |
| `use_spatial_score` | bool | true | Enable spatial scoring combining appearance, 3D distance, and recency |
| `w_app` | float | 0.55 | Weight for appearance (cosine similarity) |
| `w_dist` | float | 0.25 | Weight for 3D distance |
| `w_age` | float | 0.20 | Weight for temporal recency |
| `age_tau_sec` | float | 5.0 | Time constant for recency scoring |
| `dist_clip_m` | float | 1.0 | Distance clipping threshold in meters |
| `use_space_gating` | bool | true | Enable 3D spatial filtering |
| `gate_dist_m` | float | 0.7 | Maximum distance for spatial gating |

## 📡 ROS Interface

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/landmarks` | `landmarks_detection/landmark_array` | Detected landmarks |
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
│   ├── landmark.py              # Main ROS node
│   └── landmark_update.py       # Enhanced ROS node with re-ID
├── launch/
│   ├── landmark.launch          # Basic launch file
│   └── landmark_update.launch   # Enhanced launch file with re-ID
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
rospack find landmarks_detection
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

## 🔍 Re-Identification Algorithms

The enhanced `landmark_update.py` node implements sophisticated re-ID algorithms:

### 1. **ID Debouncing**
- Prevents rapid ID switching when landmarks are temporarily occluded
- Maintains consistent IDs across frames using spatial proximity
- Configurable debounce window and center matching radius

### 2. **Prototype Snapshots**
- Stores multiple feature vectors per landmark ID
- Improves matching robustness against viewpoint changes
- Automatically manages snapshot lifecycle with configurable limits

### 3. **Class-Specific Thresholds**
- Optimized similarity thresholds for different object types
- Accounts for intra-class vs inter-class feature variations
- Example: bottles may need lower thresholds than chairs

### 4. **Spatial Scoring**
- Combines multiple cues for robust re-ID:
  - **Appearance**: Cosine similarity of BoVW features
  - **3D Distance**: Euclidean distance in 3D space
  - **Temporal Recency**: Time-weighted scoring
- Configurable weights for each component

### 5. **3D Spatial Gating**
- Pre-filters candidates based on 3D distance
- Reduces computational load for large scenes
- Configurable distance thresholds

## 📝 Notes

- **Feature Detector**: Uses SIFT (128 dimensions) by default
- **Camera**: Designed for Intel RealSense D435i
- **Depth Processing**: Uses median filtering for robust depth estimation
- **SLAM Integration**: Compatible with Cartographer ROS
- **Performance**: Optimized for real-time operation
- **Re-ID**: Advanced algorithms for consistent landmark tracking

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
