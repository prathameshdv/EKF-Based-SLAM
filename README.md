# EKF-SLAM using Velocity Motion Model 

An **Extended Kalman Filter (EKF) SLAM** implementation in ROS2 that uses a **velocity motion model** and **LIDAR scan-based landmark detection**.  
This project demonstrates real-time robot localization and mapping by fusing **odometry** and **laser scan data** using probabilistic robotics principles.

---

## 🧠 Overview

This package implements **Simultaneous Localization and Mapping (SLAM)** using the **Extended Kalman Filter** framework.  
It estimates both:
- the **robot pose** `[x, y, yaw]`, and  
- the positions of **landmarks** detected from LIDAR data.

Key components:
- Motion prediction based on the **velocity motion model**.
- Correction using **range and bearing** landmark observations.
- **DBSCAN clustering** for grouping laser scan points.
- Visualization of robot pose, uncertainty ellipses, and landmark map using **matplotlib**.

## Algorithm Overview


### 🧩 EKF Algorithm Steps

<p align="center">
  <img src="https://github.com/prathameshdv/EKF-Based-SLAM/blob/main/EKF.png?raw=true" alt="EKF Algorithm Diagram" width="500"/>
</p>

The EKF algorithm consists of **prediction** and **update** stages as shown above.  
It estimates both the robot’s position and landmark locations while continuously refining uncertainty.

### 🌀 Velocity Motion Model

The following equation represents the **velocity-based kinematic model** used for motion prediction:

<p align="center">
  <img src="https://github.com/prathameshdv/EKF-Based-SLAM/blob/main/Velocity Model.png?raw=true" alt="Velocity Motion Model" width="500"/>
</p>

This model accounts for both **linear** and **angular velocity**, ensuring accurate pose estimation even during curved motion trajectories.

---

### Visualization

The node displays:

Blue circle: robot position.

Red ellipse: robot pose covariance

Red crosses: landmarks

Gray dots: LIDAR scan points

Heatmap: covariance matrix of the full SLAM state

The plot updates live as the robot moves.



---
