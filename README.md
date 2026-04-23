
# EKF-Fast-LIO2

This module integrates a customized Extended Kalman Filter (EKF) into the Fast-LIO2 codebase. The EKF performs sensor fusion using:

- **Fast-LIO2 odometry**
- **Wheel encoder odometry**
- **IMU data** 

The EKF is adapted from the `adaptive_filter` module of EKF-LOAM and incorporates dynamic confidence adjustment based on sensor noise profiles.

---

# Dependencies

- Livox-SDK2:

Please follow the guidance of installation in the [Livox-SDK2/README.md](https://github.com/Livox-SDK/Livox-SDK2/blob/master/README.md)

Summary:

```bash
git clone https://github.com/Livox-SDK/Livox-SDK2.git
cd ./Livox-SDK2/
mkdir build
cd build
cmake .. && make -j
sudo make install
```

- Livox ROS Driver2: (For ROS1)

Livox ROS Driver 2 is the 2nd-generation driver package used to connect LiDAR products produced by Livox, applicable for ROS (noetic recommended) and ROS2 (foxy or humble recommended). Please follow the guidance of installation in the [livox_ros_driver2](https://github.com/UFMG-Petrobras-OP-1319/livox_ros_driver2.git)

Summary for ROS1:
```bash
git clone https://github.com/Livox-SDK/livox_ros_driver2.git

source /opt/ros/noetic/setup.sh
./build.sh ROS1

```

---

## 📦 Features

- Real-time adaptive sensor fusion.
- Configurable parameters through a YAML file.
- Supports LiDAR, wheel, and IMU odometry streams.
- Lightweight implementation with extendable structure.

---

## 🚀 How to Launch

1. Ensure all required topics are being published:
    - `/Odometry` (Fast-LIO2)
    - `/wheel_odom` (wheel encoders)
    - `/imu/data` (IMU)

2. Run the launch file:
```bash
roslaunch ekf_fast_lio2 ekf_fast_lio2.launch
```

---

## 🔧 Configuration

The configuration file is located at `config/adaptive_filter_parameters.yaml`. Key parameters include:

```yaml
 # Filter settings
  enableImu: true
  enableWheel: true
  enableLidar: true
  filterFreq: "l"
  
  # Covariance gains
  lidarG: 75
  wheelG: 0.5
  imuG: 100

  # Topic names
  imuTopic: "/imu/data"
  wheelTopic: "/wheel_odom"
  FastLIO2_OdometryTopic: "/Odometry"
  filterTopic: "/filter_odom"

  # Wheel odometry covariance adaptive positive constants
  gamma_vx: 0.05
  gamma_omegaz: 0.01
  delta_vx: 0.0001
  delta_omegaz: 0.00001
```

---

## 📤 Output

- `/filter_odom` (type: `nav_msgs/Odometry`)

---

## 🔁 Feeding the filtered odometry back into the mapping process

The mapping node (`laserMapping`) can optionally consume `/filter_odom` and
use the fused LiDAR + wheel + IMU pose as the reference for the ikd-Tree map
update, instead of the Fast-LIO2 internal prediction alone.

### How it works

1. `p_imu->Process()` propagates the IKF state forward with IMU integration.
2. **If `mapping/use_filter_odom: true`**, `laserMapping` overwrites the
   `pos` and `rot` fields of the IKF state (`kf.change_x`) with the latest
   message received on `adaptive_filter/filterTopic` (default
   `/filter_odom`). Velocities, biases and gravity are kept as-is.
3. `lasermap_fov_segment()`, point undistortion, the iterated LiDAR update
   and `map_incremental()` then run on top of this fused pose.

The feedback is automatically skipped while `EKFAdaptiveFilter` has not
produced its first message yet (cold start) and whenever the message is
older than 0.5 s, so mapping keeps working even if the filter is stopped
or delayed.

### How to enable it

In the sensor YAML (`config/mid360_gazebo.yaml` or
`config/velodyne_gazebo.yaml`):

```yaml
mapping:
    # ... other mapping options ...
    use_filter_odom: true   # set to false to fall back to stock Fast-LIO2
```

The topic name is read from the adaptive-filter namespace:

```yaml
adaptive_filter:
    filterTopic: "/filter_odom"
```

No extra launch configuration is required — the existing
`ekf_fast_lio2.launch` starts both `laserMapping` and `EKFAdaptiveFilter`;
once the filter begins publishing, the mapper starts using its output
automatically.

---

## 📂 File Structure


This EKF module is merged into the existing **Fast-LIO2** directory structure as follows:

```
EKF-Fast-LIO2/
├── src/
│   └── EKFAdaptiveFilter.cpp         # EKF filter implementation
├── include/
│   └── settings_adaptive_filter.h   # EKF parameter definitions
├── config/
│   └── adaptive_filter_parameters.yaml  # YAML configuration file
├── launch/
│   └── ekf_fast_lio2.launch                   # ROS launch file (optional)
```

---

## 🧩 Dependencies

- `roscpp`
- `std_msgs`
- `geometry_msgs`
- `nav_msgs`
- `sensor_msgs`
- `tf`
- **Fast-LIO2 core dependencies**

---
