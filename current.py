import rclpy, math, cv2, os
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage

"""
cb 

unset-turtlebot
source ~/ros2_venv/bin/activate
source ~/ros2_ws/install/setup.bash
set-turtlebot 3
ros2 service call /T3/reset_pose irobot_create_msgs/srv/ResetPose {}
~/ros2_venv/bin/python3 -m tb4_sensor_reader.obstacle_avoidance

"""


NAMESPACE = '/T3' # ← change to your robot namespace
FORWARD_SPEED = 0.2 # m/s
TURN_SPEED = 0.2 # rad/s
DRIVE_TURN_SPEED = 0.25
AVOID_DISTANCE = 0.35 # metres
SIDE_AVOID_DIST = 0.15
FRONT_ARC_DEG = 45 # degrees either side of forward
TURN_180_TIME = math.pi / TURN_SPEED
DOCK_DISTANCE = 0.25
WAIT_TIME = 2.0
CUBE_MIN_DETECTION_TIME = 0.25
DEBUG_MODE = True
CHECKPOINT_FREQUENCY = 100

RED_LOW1 = np.array([0, 120, 70])
RED_HIGH1 = np.array([10, 255, 255])
RED_LOW2 = np.array([170, 120, 70])
RED_HIGH2 = np.array([180, 255, 255])

MIN_PIXELS = 2500

class ObstacleAvoidance(Node):
    def __init__(self):
        super().__init__('obstacle_avoidance')
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(
            Twist, f'{NAMESPACE}/cmd_vel', 10)
        self.scan_sub = self.create_subscription(
            LaserScan, f'{NAMESPACE}/scan',
            self.scan_callback, 10)
        self.current_x = 0.0
        self.current_y = 0.0
        self.create_subscription(
                Odometry,
                f'{NAMESPACE}/odom',
                self.odom_callback,
                10
            )
        #self.create_subscription(DockStatus, f"{NAMESPACE}/dock_status", self.dock_status, 10)
        
        self.last_turn = 'FORWARD'
        self.same_direction_counter = 0
        self.nearest_front = float('inf')
        self.nearest_left = float('inf')
        self.nearest_right = float('inf')
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('Avoidance controller started')
        self.cube_detected = False
        topic = f'{NAMESPACE}/oakd/rgb/image_raw/compressed'
        self.create_subscription(CompressedImage, topic, self.image_callback, 10)
        self.cameraMsg = CompressedImage()
        self.photoTaken = False
        self.waitElapsed = 0.0
        self.cubeDetectTime = 0.0
        self.elapsed = 0.0
        self.phase = 0
        self.phase_reading_count = 0
        self.checkpoint_x = 0.0
        self.checkpoint_y = 0.0
        self.checkpoint_yaw = None
        self.heading_turn = False
        self.turn_elapsed = 0.0
        self.turn_time = None
        self.turn_status = False
        self.current_yaw = None
        self.log_msg = None
        self.detection_x = None
        self.detection_y = None
        self.heading_target_angle = None
        

    def getYaw(self, x, y, z, w):
        """Calculates yaw in radians and degrees from a quaternion."""
        # Formula components
        numerator = 2.0 * (w * z + x * y)
        denominator = 1.0 - 2.0 * (y**2 + z**2)
        
        # Calculate yaw
        yaw_rad = math.atan2(numerator, denominator)
        
        return yaw_rad


    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_yaw = self.getYaw(msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)

    def image_callback(self, msg):
        self.cameraMsg = msg
        img = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(cv2.inRange(hsv, RED_LOW1, RED_HIGH1), cv2.inRange(hsv, RED_LOW2, RED_HIGH2))

        pixels = cv2.countNonZero(mask)
        overlay = img.copy()
        overlay[mask > 0] = [0, 0, 255]
        cv2.putText(overlay, f'[ CubeDetection ] | Red pixels: {pixels}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        
        if cv2.countNonZero(mask) >= MIN_PIXELS and self.cubeDetectTime < CUBE_MIN_DETECTION_TIME:
            self.cubeDetectTime += 0.1
            cv2.putText(overlay, '[ CubeDetection ] | DETECTED', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            self.get_logger().info(f'[ CubeDetection ] | Red cube detected — {pixels} pixels')

            if self.cubeDetectTime >= CUBE_MIN_DETECTION_TIME:
                self.cube_detected = True
            
        else:
            self.cubeDetectTime = 0.0
            self.cube_detected = False

        
        cv2.imshow('Detection', overlay)
        cv2.waitKey(1)

    def euclidean_distance_xy(self, x1, y1, x0 = 0.0, y0 = 0.0):
        return math.sqrt((x1 - x0)**2 + (y1 - y0)**2)

    def scan_callback(self, msg):
        inc = msg.angle_increment
        offsetAngle = math.radians(90)
        arc_r = math.radians(FRONT_ARC_DEG)
        side_r = math.radians(90)
        # front_i = int(round(-msg.angle_min / inc))
        front_i = 270
        half_a = int(round(arc_r / inc))
        side_a = int(round(side_r / inc))
        n = len(msg.ranges)

        def arc_min(lo, hi):
            lo = max(0, lo); hi = min(n - 1, hi)
            vals = [r for r in msg.ranges[lo:hi+1] if msg.range_min < r < msg.range_max]
            return min(vals) if vals else float('inf')

        self.nearest_front = arc_min(front_i - half_a, front_i + half_a)
        self.nearest_left = arc_min(front_i, front_i + side_a)
        self.nearest_right = arc_min(front_i - side_a, front_i)

    """
        def dock_status(self, msg):
            return msg.is_docked, msg.dock_visible
    """

    def take_photo(self, msg):
        if msg.linear.x == 0.0 and msg.angular.z == 0.0 and self.cube_detected and not self.photoTaken:
            img = self.bridge.compressed_imgmsg_to_cv2(self.cameraMsg, 'bgr8')
            img_filename = "cube_detected.png"
            cv2.imwrite(img_filename, img)
            self.get_logger().info(f"[Logging] | Cube detected. Saved to {img_filename}")
            return True
        else:
            return False

    def turn_rads(self, angle):
        self.turn_elapsed += 0.1
        msg = Twist()
        
        if not self.turn_time:
            self.turn_time = abs(angle) / TURN_SPEED
        if DEBUG_MODE:
            self.get_logger().warn(f"[Debug] | Turning | Elapsed Time: {self.turn_elapsed}, Turn Time: {self.turn_time}")
        if self.turn_elapsed <= self.turn_time:
            msg.linear.x = 0.0
            msg.angular.z = (angle / abs(angle)) * TURN_SPEED
            return msg, False
        else:
            self.turn_time = None
            msg.linear.x = 0.0
            msg.angular.z= 0.0
            self.turn_elapsed = 0.0
            return msg, True
        
    
    def check_heading(self, x0 = 0.0, y0 = 0.0, x1 = 0.0, y1 = 0.0, towards_dock = False):
        dist_dock_A = self.euclidean_distance_xy(x0,y0,0,0)
        dist_dock_B = self.euclidean_distance_xy(x1,y1,0,0)
        if towards_dock:
            if dist_dock_A < dist_dock_B:
                return True
            else:
                return False
            
        else:
            if dist_dock_A > dist_dock_B:
                return True
            else:
                return False
    
    def angle_diff(self, angle1, angle2):
        diff = angle1 - angle2
        return math.atan2(math.sin(diff), math.cos(diff))

    def heading_angle_dock(self, msg):
        target_angle = math.atan2(self.current_y, self.current_x)
        heading_err = self.angle_diff(target_angle, self.current_yaw)
        if DEBUG_MODE:
            self.get_logger().warn(f"[Debug] | >> Target Angle: {target_angle}")

        return heading_err

    def control_loop(self):
        msg = Twist()

        def navigate():

            self.get_logger().warn("-----------in Navigation ():")
            
            if self.nearest_front > AVOID_DISTANCE:
                msg.linear.x = FORWARD_SPEED
                msg.angular.z = 0.0
                self.last_turn = 'FORWARD'
                self.log = f"[No obstacle detected in {AVOID_DISTANCE}m]"

                if not(self.nearest_left < SIDE_AVOID_DIST and self.nearest_right < SIDE_AVOID_DIST):
                    
                    msg.angular.z = DRIVE_TURN_SPEED if self.nearest_left > self.nearest_right else -DRIVE_TURN_SPEED
                    self.log_msg = f"[Moving to center]"
            
            else:
                msg.linear.x = 0.0

                if self.nearest_left >= self.nearest_right:
                    if self.last_turn == 'LEFT':
                        msg.angular.z = TURN_SPEED
                        self.log_msg = "[Continuing motion]"                        
                    elif self.last_turn == 'FORWARD':
                        msg.angular.z = TURN_SPEED
                        self.last_turn = 'LEFT'
                        self.log_msg = "[Left has more clearance]"
                    else:
                        msg.angular.z = -TURN_SPEED
                        self.log_msg = "[Continuing motion]"
                
                else:
                    if self.last_turn == 'RIGHT':
                        msg.angular.z = -TURN_SPEED
                        self.log_msg = "[Continuing motion]" 
                    elif self.last_turn == 'FORWARD':
                        msg.angular.z = -TURN_SPEED
                        self.last_turn = 'RIGHT'
                        self.log_msg = "[Right has more clearance]"
                    else:
                        msg.angular.z = TURN_SPEED
                        self.log_msg = "[Continuing motion]" 

        #=======================        Phase Control       ==============================
        if self.phase == 0:
            self.checkpoint_x = self.current_x
            self.checkpoint_y = self.current_y
            self.checkpoint_yaw = self.current_yaw
            if DEBUG_MODE:
                self.get_logger().warn(f'[Debug] | Orientation: {self.current_yaw}')

            self.get_logger().warn(f"[Logging] | Checkpoint Set: ({self.checkpoint_x},{self.checkpoint_y}. Orientation(rads): {self.checkpoint_yaw})")
            self.phase = 1
            self.get_logger().warn(f"[PhaseChange] | ------------------------Phase {self.phase}------------------------")

        if self.phase == 1:
            self.phase_reading_count += 1
            navigate()
            if self.cube_detected:
                self.phase = 2
                self.phase_reading_count = 0
                self.get_logger().warn(f"[PhaseChange] | ------------------------Phase {self.phase}------------------------")
        elif self.phase == 2:
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            self.waitElapsed += 0.1

            if self.waitElapsed <= WAIT_TIME and not self.turn_status:
                self.detection_x, self.detection_y = self.current_x, self.current_y

            if self.waitElapsed > WAIT_TIME or self.turn_status:

                if self.take_photo(msg):
                    # change state
                    self.photoTaken = True  
                    self.turn_status = False        
                else:
                    self.get_logger().warn(f"[Logging] | Can't take photo")
                    msg.linear.x = 0.0
                    msg.angular.z = -TURN_SPEED / 5 if  not self.cube_detected else 0.0
                    self.turn_status = True
                    self.waitElapsed = 0.0

            if self.photoTaken:
                msg, self.turn_status = self.turn_rads(math.pi)
                self.log_msg = "[Heading away from cube]"

                if self.turn_status:
                    self.phase = 3
                    self.turn_status = False
                    self.get_logger().warn(f"[PhaseChange] | ------------------------Phase {self.phase}------------------------")

        elif self.phase == 3:
            self.phase_reading_count += 1
            if self.phase_reading_count == 1:
                self.checkpoint_x = self.current_x
                self.checkpoint_y = self.current_y
            if self.phase_reading_count % (CHECKPOINT_FREQUENCY * 2) == 0:
                self.heading_turn = self.check_heading(self.checkpoint_x, self.checkpoint_y, self.current_x, self.current_y, True)
                if not self.heading_turn:
                    self.checkpoint_x = self.current_x
                    self.checkpoint_y = self.current_y
            if self.heading_turn:
                heading_angle = self.heading_angle_dock(msg)
                msg, self.turn_status = self.turn_rads(heading_angle)
                self.log_msg = "[Correcting heading]"
                self.get_logger().warn("[Logging] | [HeadingCheck] | Turning heading, back to origin")

                if self.turn_status:
                    self.log_msg = "[Heading Corrected]"
                    self.heading_turn = False
                    self.turn_status = False
            if not self.heading_turn:
                navigate()
            if self.euclidean_distance_xy(x1=self.current_x, y1=self.current_y) < (DOCK_DISTANCE * 2):
                #is_docked, dock_visible = self.dock_status()
                #if dock_visible:
                os.system(f"ros2 action send_goal {NAMESPACE}/dock irobot_create_msgs/action/Dock {{}}")
                #if is_docked or self.euclidean_distance_xy(x1=self.current_x, y1=self.current_y) < DOCK_DISTANCE:
                self.phase = 4
                self.phase_reading_count = 0
                self.get_logger().warn(f"[PhaseChange] | ------------------------Phase {self.phase}------------------------")

        if self.phase != 4:
            if msg.angular.z == 0:
                if msg.linear.x == 0:
                    self.get_logger().info(f"Phase[{self.phase}] | [STOP] | pos: ({self.current_x:.2f}, {self.current_y:.2f}) | checkpoint: ({self.checkpoint_x}, {self.checkpoint_y}) | {self.log_msg}")
                else:
                    self.get_logger().info(f"Phase[{self.phase}] | [DRV - FWD] | pos: ({self.current_x:.2f}, {self.current_y:.2f}) | checkpoint: ({self.checkpoint_x}, {self.checkpoint_y}) | {self.log_msg}")
            else:
                if msg.linear.x == 0:
                    if msg.angular.z > 0:
                        self.get_logger().info(f"Phase[{self.phase}] | [ROT - LFT] | pos: ({self.current_x:.2f}, {self.current_y:.2f}) | checkpoint: ({self.checkpoint_x}, {self.checkpoint_y}) | {self.log_msg}")
                    else:
                        self.get_logger().info(f"Phase[{self.phase}] | [ROT - RGT] | pos: ({self.current_x:.2f}, {self.current_y:.2f}) | checkpoint: ({self.checkpoint_x}, {self.checkpoint_y}) | {self.log_msg}")
                else:
                    if msg.angular.z > 0:
                        self.get_logger().info(f"Phase[{self.phase}] | [TURN - LFT] | pos: ({self.current_x:.2f}, {self.current_y:.2f}) | checkpoint: ({self.checkpoint_x}, {self.checkpoint_y}) | {self.log_msg}")
                    else:
                        self.get_logger().info(f"Phase[{self.phase}] | [TURN - RGT] | pos: ({self.current_x:.2f}, {self.current_y:.2f}) | checkpoint: ({self.checkpoint_x}, {self.checkpoint_y}) | {self.log_msg}")

            self.publisher.publish(msg)
        
def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoidance()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
