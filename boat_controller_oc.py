from pymavlink import mavutil
from serial import SerialException
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from build.depth_wrapper import VideoProcessing 

class BoatController:
    def __init__(self, serial_port='/dev/ttyUSB0', baudrate=57600):

        # Try the serial port; if we have no connection, get depth anyway
        try:
            self.connection = mavutil.mavlink_connection(serial_port, baud=baudrate)
            self.connection.wait_heartbeat()
            print("Heartbeat Received From Boat Connection")
        except SerialException:
            print(f"\nFailed to initialize the connection at {serial_port}!\n")
            self.connection = None

        self.vid_processor = VideoProcessing()

        plt.ion()
        self.fig,self.ax = plt.subplots(1,2)
        self.drawn, = self.ax[0].plot([],[],'ro',markersize=4)
        self.heatmap = self.ax[1].imshow(np.zeros((720,1280),dtype=np.float64),cmap="plasma_r")
        circles = []
        for i in range(4):
            self.ax[0].add_patch(patches.Circle((0.0, 0.0), radius=1000*(i+1), fill=False, edgecolor='black', linewidth=1))
        self.ax[0].set_xlabel("X position (mm)")
        self.ax[0].set_ylabel("Z position (mm)")
        self.ax[0].set_xlim(-5000.0,5000.0)
        self.ax[0].set_ylim(-5000.0,5000.0)
        self.fig.canvas.draw()
        

    def pass_distances(self):
        """ 
        Pass obstacle distances as if we had a lidar 
        """
        # print("Starting pass distances")

        # We have to convert discrete points defining obstacle bbox corners to 
        # continuous obstacle present/not present vals at each given angle

        # Convert range and theta values into the shape MAVLINK wants
        # https://mavlink.io/en/messages/common.html, message 330

        # depends on support for extensions to message 330

        # print("About to grab depth")
                
        fov_deg = 110
        num_pts = 72
        num_rows = 720
        num_cols = 1280

        min_supported_dist = 300 # in mm
        max_supported_dist = 10000 # in mm

        spoof_depth_option = 0
        row_step = 1                # Row step (Increment to pixels along rows when sampling)
        col_step = 1                # Col step (Increment to pixels along cols when sampling)
        processing_flags = 4        # Processing flags
        patch_size = 7              # Patch size for bilateral filter
        sigma = 500.0               # Sigma for bilateral filter

        target_quantile = 0.01      # Quantile of depth to use per measurement

        # Theoretically we can mask out values vertically far from the camera
        y_floor = -750 # mm above the camera
        y_ceil = 750 # mm below the camera
        
        t0 = time.perf_counter()
        if spoof_depth_option == 1:
            ret = 0
            pts = np.zeros(shape=(num_rows,num_cols,3))
            r = np.array(range(num_rows))
            c = np.array(range(num_cols))
            C,R = np.meshgrid(c,r)
            pts[:,:,2] = 2000 + 1*C + 1*R
            pts[:,:,1] = (R - 360)*5
            pts[:,:,0] = (C - 640)*5
        else:
            ret,pts = self.vid_processor.grab_depth(
                row_step,
                col_step,              
                processing_flags,              
                patch_size,              
                sigma           
            )
        
        if ret != 0:
            return
        
        t1 = time.perf_counter()
        print(f"Time to grab depth: {t1 - t0}")
        # print(f"Original shape: {pts.shape}")

        t_heatmap = time.perf_counter()
        mask = ~np.isnan(pts).any(axis=2)

        pts_zeroed = pts.copy()
        pts_zeroed[~mask] = 8000.0 # set Nan values to 8m
        pts_zeroed = pts_zeroed[:,:,2] # get only the Z (depth)
        # print(f"Shape of pts zeroed: {pts_zeroed.shape}")
        # print(f"pts zeroed: {pts_zeroed}")

        grid_indices = np.indices((num_rows,num_cols))
        r = np.array(range(num_rows))
        c = np.array(range(num_cols))
        C,R = np.meshgrid(c,r)
        row_cond = (grid_indices[0] % row_step == 0)
        col_cond = (grid_indices[1] % col_step == 0)
        cond = row_cond & col_cond
        pts_zeroed = pts_zeroed[cond].reshape(num_rows // row_step,num_cols // col_step)
        # print(f"Resized points zeroed shape: {pts_zeroed.shape}")

        self.heatmap.set_data(pts_zeroed)
        #self.heatmap.set_clim(pts_zeroed.min(), pts_zeroed.max())
        self.heatmap.set_clim(min_supported_dist,5000)
        print(f"Time to update heatmap: {time.perf_counter() - t_heatmap}")

        pts = pts[mask]
        # print(f"Grabbed depth: ret: {ret}, pts: {pts}")
        # print(f"Final shape: {pts.shape}")
        # print(f"Min raw dist: {min(np.linalg.norm(pts,axis=1))}")
        # print(f"Max x dist: {max(pts[:,0])}")
        # print(f"Max y dist: {max(pts[:,1])}")
        # print(f"Max z dist: {max(pts[:,2])}")

        # Mask out too far y values
        y_valid_mask = (pts[:,1] < y_ceil) & (pts[:,1] > y_floor)
        pts = pts[y_valid_mask]
        print(f"Masked out {sum(~y_valid_mask)} pts outside the Y range, out of {len(y_valid_mask)} pts")
        
        t_yaw = time.perf_counter()
        yaws = np.atan2(pts[:,0],pts[:,2]) * 180 / np.pi
        print(f"Time to compute yaws: {time.perf_counter() - t_yaw}")
        # print(f"Yaws shape: {yaws.shape}")
        # print(f"Yaw min and max: {max(yaws)}, {min(yaws)}")

        angle_offset = float(fov_deg/2) # CW from forward
        increment_f = -1*fov_deg/num_pts
        bin_width = fov_deg / num_pts

        frame = mavutil.mavlink.MAV_FRAME_BODY_FRD

        depths_passed = [max_supported_dist+1]*72
        angles_passed = [angle_offset + increment_f*i for i in range(72)]

        t0_angles = time.perf_counter()
        bin_indices = np.floor((angle_offset - yaws) / bin_width).astype(int)
        bin_indices = np.clip(bin_indices, 0, num_pts-1)
        # print(f"Max, mean, min bin indices {np.max(bin_indices)}, {np.mean(bin_indices)}, {np.min(bin_indices)}")

        target_quantiles = [[] for i in range(72)]
        depth = np.hypot(pts[:,0], pts[:,2])
        depths_passed = np.full(num_pts, max_supported_dist+1, dtype=float)
        for b, d in zip(bin_indices, depth):
            if d < depths_passed[b]:
                depths_passed[b] = d
            target_quantiles[b].append(d)

        tq = time.perf_counter()
        target_quantiles = [np.array([max_supported_dist+1]) if len(list_) == 0 else np.array(list_) for list_ in target_quantiles]
        depths_passed = np.full(num_pts, max_supported_dist+1, dtype=float)
        for i in range(len(depths_passed)):
            depths_passed[i] = np.quantile(target_quantiles[i],target_quantile)
        #print(f"Time to do quantiles: {time.perf_counter() - tq}")

        print(f"Time to get angles closest: {time.perf_counter() - t0_angles}")
        #print(f"depth shape {depth.shape}")

        #print(f"Angles passed: {angles_passed}")
        #print(f"Depths passed has shape {depths_passed.shape} and values: {depths_passed}")
        
        # Plotting code --- just for visualization
        x_obs = [depths_passed[i] * np.cos(angles_passed[i] * np.pi / 180) for i in range(len(angles_passed)) if depths_passed[i] <= max_supported_dist]
        y_obs = [depths_passed[i] * np.sin(angles_passed[i] * np.pi / 180) for i in range(len(angles_passed)) if depths_passed[i] <= max_supported_dist]
        #print(f"X obs, Y obs: {x_obs}, {y_obs}")
        #print(f"Min depth: {min([np.sqrt(x_obs[i]**2 + y_obs[i]**2) for i in range(len(x_obs))])}")
        
        ttt = time.perf_counter()
        self.drawn.set_xdata(y_obs)
        self.drawn.set_ydata(x_obs)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        print(f"Drawing took {time.perf_counter() - ttt}")
        
        if self.connection is not None:
            self.connection.mav.obstacle_distance_send( 
                int(time.time() * 1e6), # time in us
                0,                      # sensor type
                depths_passed, 
                1,  # incrememnt (unused if given increment_f)
                min_supported_dist, 
                max_supported_dist,
                increment_f,
                angle_offset,
                frame
            )
        print(f"\n\nIteration time: {time.perf_counter() - t0}\n")

    def arm(self):
        """
        Arms boat.
        """
        print("Arming boat...")
        self.connection.mav.command_long_send(
        self.connection.target_system,
        self.connection.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1, 0, 0, 0, 0, 0, 0)

        self.connection.motors_armed_wait()
        print("ARMED!")

    def navigate_to_waypoint(self, lat, lon, alt):
        print(f"Navigation to Waypoint: {lat}, {lon}, {alt}")
        self.connection.mav.mission_item_send(
            self.connection.target_system,
            self.connection.target_component,
            0,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            0, 1, 0, 0, 0, 0, lat, lon, alt
        )
    
    def wait_for_waypoint(self, lat, lon, alt):
        while True:
            msg = self.connection.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
            if msg:
                current_lat = msg.lat / 1e7
                current_lon = msg.lon / 1e7
                current_alt = msg.relative_alt / 1000
                tolerance = 1.0

                distance_to_waypoint = self.calculate_distance(lat, lon, current_lat, current_lon)
                print(f"Current Position: {current_lat}, {current_lon}, Altitude: {current_alt}m")

                if distance_to_waypoint < tolerance: # in meters
                    print("Reached waypoint, holding Position.")
                    self.hold_position(current_lat, current_lon, current_alt)
                    break

    
    def hold_position(self, lat, lon, alt):
        print(f"Holding position at {lat}, {lon}, {alt} m")
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_CMD_DO_REPOSITION,
            0, 0, 0, 0, 0, lat, lon, alt
        )
    
    def return_to_home(self):
        print("Returning to home...")
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
            0, 0, 0, 0, 0, 0, 0, 0
        )
        self.monitor_home_return()

    def monitor_home_return(self):
        home_lat, home_lon = None

        while True:
            msg = self.connection.recv_match(type='HOME_POSITION', blocking=True)
            if msg:
                home_lat = msg.latitude / 1e7
                home_lon = msg.longitude / 1e7
                print(f"Home Location: {home_lat}, {home_lon}")
                break
        
        self.wait_for_waypoint(home_lat, home_lon)

def main():
    controller = BoatController()
    print("Created controller")
    while True:
        controller.pass_distances()

if __name__ == "__main__":
    main()