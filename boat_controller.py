from pymavlink import mavutil
import numpy as np
import cv2
import time
from get_obstacle_dist import DistanceProcess

class BoatController:
    def __init__(self, connection, serial_port='/dev/ttyUSB0', baudrate=57600):
        #self.connection = mavutil.mavlink_connection(serial_port, baud=baudrate)
        #self.connection.wait_heartbeat()
        self.connection = connection
        self.img_processor = DistanceProcess()
        print("Heartbeat Received From Boat Connection")

    def pass_distances(self, connection):
        ranges,angles,angular_lengths,obstacle_dists = self.img_processor.get_dist()
        angles_deg = [x*np.pi / 180 for x in angles]

        # We have to convert discrete points defining obstacle bbox corners to 
        # continuous obstacle present/not present vals at each given angle

        # Convert range and theta values into the shape MAVLINK wants
        # https://mavlink.io/en/messages/common.html, message 330

        # depends on support for extensions to message 330

        fov_deg = 110
        num_pts = 72

        min_supported_dist = 1 # in cm
        max_supported_dist = 1000 # in cm

        angle_offset = float(fov_deg/2) # CW from forward
        increment_f = -1*fov_deg/num_pts
        frame = mavutil.mavlink.MAV_FRAME_BODY_FRD

        distances = []
        angles_passed = [angle_offset + increment_f*i for i in range(72)]

        for angle in angles_passed:
            min_dist = None
            found_obstacle = False
            for idx,obs_angles in enumerate(angular_lengths):
                if min(obs_angles) < angle < max(obs_angles):
                    found_obstacle = True
                    if not min_dist or min_dist > min(obstacle_dists[idx]):
                        min_dist = min(obstacle_dists[idx])
            if not found_obstacle:
                distances.append(max_supported_dist + 1)
            else:
                distances.append(min_dist)

        connection.mav.obstacle_distance_send( 
            int(time.time() * 1e6), # time in us
            0,                      # sensor type
            distances, 
            1,  # incrememnt (unused if given increment_f)
            min_supported_dist, 
            max_supported_dist,
            increment_f,
            angle_offset,
            frame
        )

        pass

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