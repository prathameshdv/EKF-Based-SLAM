#!/usr/bin/python3

import numpy as np
from easydict import EasyDict as edict
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.cluster import DBSCAN
from .util import quaternion_to_yaw, get_center_radius,\
                  angle_between_yaw, plot_cov

class SlamEkf(Node):
    def __init__(self):
        super().__init__('ekf_slam_velocity_model')

        # for grouping lidar scan points
        self.dbscan = DBSCAN(eps=0.2,min_samples=1)
        
        # subscriber to odom topic, use twist as control input
        self.odom_sub = self.create_subscription(Odometry, '/odom',
                                                 self.velocity_callback,10)
        
        # Control input u = [v, w]
        self.control_input = np.zeros((2, 1))

        # subscriber to laser scan
        self.scan_sub = self.create_subscription(LaserScan, "/scan",
            self.scan_callback,qos_profile=qos_profile_sensor_data)
        
        self.landmark_measurements = edict(data=None,timestamp=None)
        self.lidar_pts_fixedframe = None
        
        cb_group = MutuallyExclusiveCallbackGroup()
        # create a timer for SLAM updates
        self.create_timer(0.01, self.slam,
                          callback_group=cb_group)
        self.slam_last_ts = None
        
        # initialize robot pose state belief
        # the landmarks are added as the robot observes them
        self.mu = np.zeros((3,1))
        self.sigma = np.zeros((3,3))

        # coefficient for motion noise covariance (velocity model)
        # Corresponds to noise from: v->v, w->v, v->w, w->w
        self.alpha = [0.01, 0.001, 0.001, 0.01]
        
        # coefficient for the sensor noise variance
        self.beta = [0.01,0.001]

        self.landmark_count = 0

        # create plot to show the slam process
        self.fig = plt.figure(figsize=(17,10),constrained_layout=True)
        ncol = 3
        gs = GridSpec(2, ncol, figure=self.fig)
        self.ax1 = self.fig.add_subplot(gs[:, :(ncol-1)],frameon=False)
        self.ax1.set_aspect('equal')
        self.ax2 = self.fig.add_subplot(gs[0, ncol-1],frameon=False)
        self.ax2.set_aspect('equal')
        self.fig.show()
        
        # store handles for all the plots
        self.robot_plot = None
        self.robot_cov_plot = None
        self.landmark_plot = edict()
        self.landmark_measurement_plot = None
        self.lidar_plot = None
        self.cov_mat_plot = None
        self.odom_plot = None
        self.robot_traj_slam = None
        self.robot_traj_odom = None
        self.latest_odom_pose = [0., 0.] # For plotting odom path
        # timer for updating plot
        self.create_timer(0.1, self.plot_callback)

    @property
    def robot_state(self):
        return self.mu[:3]
    
    def update_robot_state(self, new_state):
        self.mu[:3] = new_state
    
    @property
    def landmark_state(self):
        return self.mu[3:].reshape((-1,2))
    
    @property
    def sigma_r(self):
        return self.sigma[:3,:3]
    
    @property
    def sigma_rm(self):
        return self.sigma[:3,3:]
    
    @property
    def sigma_mr(self):
        return self.sigma[3:,:3]
    
    @property
    def sigma_m(self):
        return self.sigma[3:,3:]
    
    def velocity_callback(self, msg : Odometry):
        """
        Callback for the odom topic. Extracts linear and angular velocity
        to use as the control input u = [v, w].
        """
        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z
        self.control_input = np.array([[v], [w]])
        
        # Also store latest pose from odom for plotting comparison
        self.latest_odom_pose[0] = msg.pose.pose.position.x
        self.latest_odom_pose[1] = msg.pose.pose.position.y

    def scan_callback(self, msg : LaserScan):
        # ... (This function remains unchanged)
        # get angles of the scan
        nscan = round((msg.angle_max - msg.angle_min)/msg.angle_increment)+1
        angles = np.linspace(msg.angle_min, msg.angle_max, nscan, endpoint=True)

        # convert to fixed frame
        x,y,yaw = self.robot_state[:,0]
        rng = np.array(msg.ranges)
        keep = (rng>=msg.range_min) & (rng<=msg.range_max)
        pt_x = x + rng[keep] * np.cos(yaw + angles[keep])
        pt_y = y + rng[keep] * np.sin(yaw + angles[keep])
        pts = np.hstack([pt_x[:,np.newaxis],pt_y[:,np.newaxis]])
        self.lidar_pts_fixedframe = pts # store for plotting

        # cluster points
        pts_clu = self.cluster_pts(pts)
        
        # find the center and radius for each cluster
        all_center = []
        for clu in pts_clu:
            center, r = get_center_radius(clu)
            if center is not None:
                all_center.append(center)
        if len(all_center)==0:
            return
        all_center = np.array(all_center)

        # convert back to range and bearing
        data = []
        for center in all_center:
            dx = center[0]-x
            dy = center[1]-y
            # range and bearing, expressed in robot frame
            angle = angle_between_yaw(yaw, np.arctan2(dy,dx))
            data.append([np.sqrt(dx**2+dy**2),angle])
        data = np.array(data)
        self.landmark_measurements.data = data
        self.landmark_measurements.timestamp = self.get_curr_time()

    def plot_callback(self):
        # ... (This function remains unchanged, except for odom plot)
        # plot robot pose
        x,y,yaw = self.robot_state[:,0]
        d = 0.3
        if self.robot_plot is None:
            # set up new plot
            self.robot_plot = [self.ax1.plot(x,y,label='robot',
                                        ms=6,color='b',marker='o',ls='')[0],
                               self.ax1.plot([x,x+d*np.cos(yaw)],
                                        [y,y+d*np.sin(yaw)],
                                        color='r')[0], # x axis, body frame
                               self.ax1.plot([x,x+d*np.cos(yaw+np.pi/2)],
                                        [y,y+d*np.sin(yaw+np.pi/2)],
                                        color='g')[0] # y axis
                            ]
        else:
            # update data only
            self.robot_plot[0].set_data(x,y)
            self.robot_plot[1].set_data([x,x+d*np.cos(yaw)],
                                        [y,y+d*np.sin(yaw)])
            self.robot_plot[2].set_data([x,x+d*np.cos(yaw+np.pi/2)],
                                        [y,y+d*np.sin(yaw+np.pi/2)])

        # robot pose cov
        self.robot_cov_plot = plot_cov(plot_handle=self.robot_cov_plot,
                                       ax=self.ax1,
                                       mu=self.robot_state[:2,0],
                                       cov=self.sigma[:2,:2])

        # landmark lidar scans
        if self.lidar_plot is None:
            if self.lidar_pts_fixedframe is not None and \
                len(self.lidar_pts_fixedframe)>0:
                self.lidar_plot = self.ax1.plot(self.lidar_pts_fixedframe[:,0],
                                        self.lidar_pts_fixedframe[:,1],
                                        label='lidar',
                                        ms=1,
                                        color=(0.8,0.8,0.8),
                                        marker='o',
                                        ls='')[0]
        else:
            self.lidar_plot.set_data(self.lidar_pts_fixedframe[:,0],
                                     self.lidar_pts_fixedframe[:,1])
            
        # landmarks
        for k,l in enumerate(self.landmark_state):
            indices = [3+2*k,3+2*k+1]
            mu = self.landmark_state[k,:]
            sigma = self.sigma[np.ix_(indices,indices)]
            name = f'landmark_{k:d}'
            if name not in self.landmark_plot:
                p1 = self.ax1.plot(l[0],l[1],
                            ms=6,color='r',marker='+',ls='')[0]
                p2 = plot_cov(plot_handle=None,
                              ax=self.ax1,
                              mu=mu,
                              cov=sigma)
                self.landmark_plot[name] = [p1,p2]
            else:
                # update
                self.landmark_plot[name][0].set_data(l[0],l[1])
                self.landmark_plot[name][1] = plot_cov(
                    plot_handle=self.landmark_plot[name][1],
                    ax=self.ax1,
                    mu=mu,
                    cov=sigma)
        
        # plot odom coordinates
        odom_x, odom_y = self.latest_odom_pose
        if self.odom_plot is None:
            self.odom_plot = self.ax1.plot(odom_x,odom_y,label='robot_odom',
                        ms=6,color='b',marker='s',ls='',mfc='none')[0]
        else:
            self.odom_plot.set_data(odom_x,odom_y)

        if self.cov_mat_plot is None:
            plt.sca(self.ax2)
            self.cov_mat_plot = plt.imshow(self.sigma, cmap='cool')
            plt.colorbar(ax=self.ax2,aspect=20,shrink=0.3)
        else:
            m = self.sigma.shape[0]
            self.cov_mat_plot.set(data=self.sigma,extent=(0,m+1,0,m+1))

        self.ax1.set(xlim=(-3,15),ylim=(-15,3))
        self.fig.canvas.flush_events()
        self.fig.canvas.draw()

    def cluster_pts(self, pts):
        # ... (This function remains unchanged)
        if len(pts)==0:
            return []
        clu = self.dbscan.fit(pts)
        labels = clu.labels_
        nclu = np.max(labels)
        result = []
        for i in range(nclu+1):
            result.append(pts[labels==i])
        return result

    def get_curr_time(self):
        # ... (This function remains unchanged)
        sec, nanosec = self.get_clock().now().seconds_nanoseconds()
        return float(sec) + float(nanosec)/1.0e9
    
    def slam_timing(self):
        # ... (This function remains unchanged)
        timeout_dur = 0.21 # duration for checking scan data availability
        curr_ts = self.get_curr_time()
        if self.slam_last_ts is None:
            # initialize timestamp
            self.slam_last_ts = self.get_curr_time()

        pred_only = True # default value
        proceed = True
        dt = curr_ts - self.slam_last_ts
        use_own_timing = True
        landmark_measurements = self.landmark_measurements
        if landmark_measurements.timestamp is not None:
            # have received scan data before
            if landmark_measurements.timestamp > self.slam_last_ts+1e-3:
                # new sensor data
                pred_only = False
                use_own_timing = False
                self.slam_last_ts = landmark_measurements.timestamp
            else:
                if curr_ts >= self.slam_last_ts + timeout_dur:
                    # passed timeout period, no new data
                    use_own_timing = True
                else:
                    # within wait period
                    proceed = False
                    return proceed, pred_only, dt
        else: # no sensor data ever
            use_own_timing = True
        
        if use_own_timing:
            # no new sensor data
            if curr_ts - self.slam_last_ts >= 0.2:
                # update around 5 Hz
                pred_only = True
                self.slam_last_ts = curr_ts
            else:
                proceed = False
                return proceed, pred_only, dt
        return proceed, pred_only, dt

    def motion_model(self, x, u, dt):
        """
        Velocity-based motion model.
        Args:
            x (np.array): Robot state [x, y, yaw]
            u (np.array): Control input [v, w]
            dt (float): Time delta
        Returns:
            np.array: Predicted robot state
        """
        yaw = x[2, 0]
        v = u[0, 0]
        w = u[1, 0]

        # Handle straight line motion to avoid division by zero
        if abs(w) < 1e-4:
            # Straight line motion
            dx = v * dt * np.cos(yaw)
            dy = v * dt * np.sin(yaw)
            d_yaw = 0
        else:
            # Circular motion
            r = v / w
            dx = -r * np.sin(yaw) + r * np.sin(yaw + w * dt)
            dy = r * np.cos(yaw) - r * np.cos(yaw + w * dt)
            d_yaw = w * dt

        return x + np.array([[dx], [dy], [d_yaw]])

    def motion_jacobian(self, u, dt):
        """
        The jacobian of state transition w.r.t previous state (Gt).
        Args:
            u (np.array): Control input [v, w]
            dt (float): Time delta
        Returns:
            np.array: 3x3 Jacobian matrix
        """
        yaw = self.robot_state[2, 0]
        v = u[0, 0]
        w = u[1, 0]

        if abs(w) < 1e-4:
            # Jacobian for straight line motion
            g13 = -v * dt * np.sin(yaw)
            g23 = v * dt * np.cos(yaw)
        else:
            # Jacobian for circular motion
            r = v / w
            g13 = -r * np.cos(yaw) + r * np.cos(yaw + w * dt)
            g23 = -r * np.sin(yaw) + r * np.sin(yaw + w * dt)

        G = np.array([[1.0, 0.0, g13],
                      [0.0, 1.0, g23],
                      [0.0, 0.0, 1.0]])
        return G

    def noise_jacobian(self, u, dt):
        """
        Computes the motion noise covariance in state space.
        Args:
            u (np.array): Control input [v, w]
            dt (float): Time delta
        Returns:
            np.array: 3x3 state space motion noise covariance matrix R
        """
        yaw = self.robot_state[2, 0]
        v = u[0, 0]
        w = u[1, 0]
        
        # Covariance of control inputs (M_t in Probabilistic Robotics)
        # Variance proportional to magnitude of control signals
        M = np.array([[self.alpha[0]*abs(v) + self.alpha[1]*abs(w), 0.0],
                      [0.0, self.alpha[2]*abs(v) + self.alpha[3]*abs(w)]])

        # Jacobian of motion model w.r.t control inputs (V_t)
        if abs(w) < 1e-4:
            # Straight line motion case
            v11 = dt * np.cos(yaw)
            v21 = dt * np.sin(yaw)
            v12 = -0.5 * v * dt**2 * np.sin(yaw)
            v22 = 0.5 * v * dt**2 * np.cos(yaw)
        else:
            # Circular motion case
            syaw = np.sin(yaw)
            cyaw = np.cos(yaw)
            syaw_dt = np.sin(yaw + w * dt)
            cyaw_dt = np.cos(yaw + w * dt)
            
            v11 = (-syaw + syaw_dt) / w
            v21 = (cyaw - cyaw_dt) / w
            v12 = (v * (syaw - syaw_dt)) / w**2 + (v * cyaw_dt * dt) / w
            v22 = (-v * (cyaw - cyaw_dt)) / w**2 + (v * syaw_dt * dt) / w

        V = np.array([[v11, v12],
                      [v21, v22],
                      [0.0, dt]])

        # Motion noise in state space
        R = V @ M @ V.T
        return R
    
    def compute_cov_pred(self, J_motion, R_motion):
        """
        Computes the covariance matrix of the predicted belief
        Args:
            J_motion (np.ndarray): Motion model jacobian w.r.t state (G)
            R_motion (np.ndarray): Motion noise covariance in state space (R)
        """
        # Update robot pose covariance
        self.sigma[:3,:3] = J_motion @ self.sigma_r @ J_motion.T + R_motion

        if self.sigma.shape[0]>3: # landmarks have been observed
            # Update robot-landmark covariance
            tmp = self.sigma_mr @ J_motion.T
            self.sigma[3:,:3] = tmp
            self.sigma[:3,3:] = tmp.T
            
    # --- The following functions are part of the CORRECTION step ---
    # --- They are independent of the motion model and remain unchanged ---

    def convert_to_fixed_frame(self, landmark_measurements):
        x,y,yaw = self.robot_state[:,0]
        landmark_xy = []
        for p in landmark_measurements:
            x_ = x + p[0]*np.cos(yaw + p[1])
            y_ = y + p[0]*np.sin(yaw + p[1])
            landmark_xy.append([x_,y_])
        return np.array(landmark_xy)

    def find_association(self, l):
        for j, ls in enumerate(self.landmark_state):
            if np.linalg.norm(ls-l)<1.0: 
                return j
        return -1
    
    def sensor_cov(self, r):
        b1, b2 = self.beta
        return np.array([[b1*r,.0],[.0, b2]])

    def initialize_landmark(self, l_xy, measurement):
        rng, bearing = measurement
        self.mu = np.vstack((self.mu, l_xy.reshape((2,1))))

        x,y,yaw = self.robot_state[:,0]
        t = yaw+bearing
        c = np.cos(t)
        s = np.sin(t)
        J1 = np.array([[1.,0.,-rng*s],
                       [0.,1., rng*c]])
        
        J2 = np.array([[c, -rng*s],
                       [s,  rng*c]])
        
        Rs = self.sensor_cov(rng)
        
        cov_ll = np.linalg.multi_dot((J1,self.sigma_r,J1.T)) + \
                 np.linalg.multi_dot((J2,Rs,J2.T))
        cov_lx = np.linalg.multi_dot((J1, self.sigma[:3,:]))

        self.sigma = np.vstack((self.sigma, cov_lx))
        self.sigma = np.hstack((self.sigma, np.vstack((cov_lx.T,cov_ll))))

        self.landmark_count+=1

        self.get_logger().info(
             (f"✅ New landmark added at ({l_xy[0]:.02f},{l_xy[1]:.02f})."
             f"Current total landmark number: {self.landmark_count}"
            ))

    def compute_obs(self, landmark_ind, rng):
        j = landmark_ind
        indices = [0,1,2,3+2*j,3+2*j+1]
        cov_ = self.sigma[np.ix_(indices,indices)]

        xr,yr,yaw = self.robot_state[:,0]
        xl,yl = self.landmark_state[j,:]
        dx, dy = (xl-xr), (yl-yr)

        rho = dx**2+dy**2
        
        H = np.array([[-dx/np.sqrt(rho), -dy/np.sqrt(rho), 0, dx/np.sqrt(rho), dy/np.sqrt(rho)],
                      [dy/rho, -dx/rho, -1, -dy/rho, dx/rho]])

        Z = np.linalg.multi_dot((H, cov_, H.T)) + self.sensor_cov(rng)
        z_pred = np.array([np.sqrt(rho), np.arctan2(dy,dx)-yaw])
        z_pred[1] = angle_between_yaw(yaw1=0, yaw2=z_pred[1]) # Normalize angle

        return H, Z, z_pred

    def slam(self):
        """perform actual slam updates
        """
        proceed, pred_only, dt = self.slam_timing()
        if not proceed:
            return
        
        # Get control input u = [v, w]
        u = self.control_input

        # ==================  1. PREDICTION STEP ==============================
        
        # State transition jacobian w.r.t state (Gt)
        G = self.motion_jacobian(u=u, dt=dt)
        
        # Motion noise covariance in state space (Rt)
        R = self.noise_jacobian(u=u, dt=dt)
        
        # Predict the mean of the robot state
        self.update_robot_state(
            self.motion_model(x=self.robot_state, u=u, dt=dt)
        )
        
        # Predict the covariance
        self.compute_cov_pred(G, R)

        if pred_only is True:
            return

        # ==================  2. CORRECTION STEP ==============================
        l_measurement_polar = self.landmark_measurements.data
        l_measurement_xy = self.convert_to_fixed_frame(l_measurement_polar)

        for z, l_xy in zip(l_measurement_polar, l_measurement_xy):
            j = self.find_association(l_xy)

            if j==-1:
                self.initialize_landmark(l_xy,measurement=z)
                j = self.landmark_count-1

            H, Z, z_pred = self.compute_obs(landmark_ind=j, rng=z[0])

            ind = [0,1,2,3+2*j,3+2*j+1]
            cov_ = self.sigma[:,ind]
            K = cov_ @ H.T @ np.linalg.inv(Z)
            
            innovation = (z - z_pred)
            innovation[1] = angle_between_yaw(yaw1=z_pred[1], yaw2=z[1]) # Normalize angle
            innovation = innovation[:,np.newaxis]

            self.mu = self.mu + K @ innovation
            self.sigma = self.sigma - K @ Z @ K.T

def main(args=None):
    rclpy.init(args=args)
    ekf_slam = SlamEkf()
    ekf_slam.get_logger().info("EKF SLAM (Velocity Model) started brother!")
    rclpy.spin(ekf_slam)
    rclpy.shutdown()

if __name__ == "__main__":
    main()

