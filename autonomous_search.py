import rclpy, math, cv2, os
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage

"""
ros2 service call /T22/reset_pose irobot_create_msgs/srv/ResetPose {}

source ~/ros2_venv/bin/activate
source ~/ros2_ws/install/setup.bash
~/ros2_venv/bin/python3 -m tb4_sensor_reader.autonomous_search

"""


NAMESPACE = '/T22' # ← change to your robot namespace
FORWARD_SPEED = 0.15 # m/s
TURN_SPEED = 0.35 # rad/s
DRIVE_TURN_SPEED = 0.3
AVOID_DISTANCE = 0.6 # metres
SIDE_AVOID_DIST = 0.15
FRONT_ARC_DEG = 45 # degrees either side of forward
TURN_180_TIME = math.pi / TURN_SPEED
DOCK_DISTANCE = 0.3
WAIT_TIME = 2.0
CUBE_MIN_DETECTION_TIME = 0.5
DEBUG_MODE = True
MAX_HEADING_ERR = 0.12

RED_LOW1 = np.array([0, 120, 70])
RED_HIGH1 = np.array([10, 255, 255])
RED_LOW2 = np.array([170, 120, 70])
RED_HIGH2 = np.array([180, 255, 255])

MIN_PIXELS = 2500

class AutonomousSearch(Node):
    def __init__(self):
        super().__init__('autonomous_search')
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(
            Twist, f'{NAMESPACE}/cmd_vel', 10)
        self.scan_sub = self.create_subscription(
            LaserScan, f'{NAMESPACE}/scan',
            self.scan_callback, 10)
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.create_subscription(
                Odometry,
                f'{NAMESPACE}/odom',
                self.odom_callback,
                10
            )
        
        self.last_turn = 'FORWARD'
        self.same_direction_counter = 0
        self.nearest_front = float('inf')
        self.nearest_left = float('inf')
        self.nearest_right = float('inf')
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('Autonomous search started')
        self.cube_detected = False
        topic = f'{NAMESPACE}/oakd/rgb/image_raw/compressed'
        self.create_subscription(CompressedImage, topic, self.image_callback, 10)
        self.cameraMsg = CompressedImage()
        self.photoTaken = False
        self.cubeDetectTime = 0.0
        self.elapsed = 0.0
        self.state = "SEARCH"
        self.returnTurn = False

    def getYaw(self, x, y, z, w):
        num = 2.0 * (w * z + x * y)
        den = 1.0 - 2.0 * (y**2 + z**2)

        yaw_rad = math.atan2(num, den)

        return yaw_rad

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_yaw = self.getYaw(msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)

  #=================================================================================  
    def image_callback(self, msg):
        self.cameraMsg = msg
        img = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(cv2.inRange(hsv, RED_LOW1, RED_HIGH1), cv2.inRange(hsv, RED_LOW2, RED_HIGH2))

        pixels = cv2.countNonZero(mask)
        overlay = img.copy()
        overlay[mask > 0] = [0, 0, 255]
        cv2.putText(overlay, f'Red pixels: {pixels}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if cv2.countNonZero(mask) >= MIN_PIXELS and self.cubeDetectTime < CUBE_MIN_DETECTION_TIME:
            self.cubeDetectTime += 0.1
            cv2.putText(overlay, 'DETECTED', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            self.get_logger().info(f'Red cube detected — {pixels} pixels')

            if self.cubeDetectTime >= CUBE_MIN_DETECTION_TIME:
                self.cube_detected = True
            
        else:
            self.cubeDetectTime = 0.0
            self.cube_detected = False

        
        cv2.imshow('Detection', overlay)
        cv2.waitKey(1)
#=================================================================================

    def euclidean_distance_xy(self, x1, y1, x0 = 0.0, y0 = 0.0):
        return math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
#=================================================================================

    def scan_callback(self, msg):
        inc = msg.angle_increment
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

        #self.get_logger().info(str(msg.angle_min))
        # self.get_logger().info(f"front_i: {front_i}")

#=================================================================================
    def take_photo(self, msg):
        if msg.linear.x == 0.0 and msg.angular.z == 0.0 and self.cube_detected and not self.photoTaken:
            img = self.bridge.compressed_imgmsg_to_cv2(self.cameraMsg, 'bgr8')
            img_filename = "cube_detected.png"
            cv2.imwrite(img_filename, img)
            self.get_logger().info(f"Cube detected. Saved to {img_filename}")
            return True
        else:

            self.get_logger().info("Can't take photo")
            return False

#====================================================#
    def control_loop(self):
        msg = Twist()

        match self.state:
            
            case "SEARCH":
                msg = self.search(msg)

                if self.cube_detected:
                    msg = self.stop(msg)
                    self.state = "REPORT"

            case "REPORT":
                if self.wait():
                    if self.report(msg):
                        self.state = "RETURN"
                        self.elapsed = 0.0
                else:
                    self.elapsed += 0.1

            case "RETURN":
                if not self.returnTurn:
                    msg, self.returnTurn = self.turnAngle(msg, 180)
                    self.elapsed += 0.1

                else:
                    self.elapsed = 0.0
                    msg = self.returnToOrigin(msg)

            case "DONE":
                pass
        
        self.publisher.publish(msg)
        self.get_logger().info(f"State: {self.state}")
        self.get_logger().info(f"Cube detected: {self.cube_detected}")

#=========================================================
    def avoid(self, msg):

        if self.nearest_front > (AVOID_DISTANCE / 2):
            msg.linear.x = FORWARD_SPEED
        
            if not(self.nearest_left < SIDE_AVOID_DIST and self.nearest_right < SIDE_AVOID_DIST):
                    msg.angular.z = DRIVE_TURN_SPEED if self.nearest_left > self.nearest_right else -DRIVE_TURN_SPEED
        else:
            self.get_logger().info(f"Obstacle in front: {self.nearest_front}")
            msg.linear.x = 0.0

            if self.nearest_left >= self.nearest_right:
                if self.last_turn == 'LEFT':
                    msg.angular.z = TURN_SPEED
                            
                elif self.last_turn == 'FORWARD':
                    msg.angular.z = TURN_SPEED
                    self.last_turn = 'LEFT'
                else:
                    msg.angular.z = -TURN_SPEED
                    
            else:
                if self.last_turn == 'RIGHT':
                    msg.angular.z = -TURN_SPEED
                elif self.last_turn == 'FORWARD':
                    msg.angular.z = -TURN_SPEED
                    self.last_turn = 'RIGHT'
                else:
                    msg.angular.z = TURN_SPEED

        return msg      
     
    def report(self, msg):
        # log current position 
        return self.take_photo(msg)

    def returnToOrigin(self, msg):

        if self.euclidean_distance_xy(0.0, 0.0, self.current_x, self.current_y) > 0.3:
            self.get_logger().info("Returning to origin")
            
            if self.nearest_front > AVOID_DISTANCE:
                msg.linear.x = FORWARD_SPEED
                msg.angular.z = 0.0
                self.last_turn = 'FORWARD'

                msg = self.find_return_heading(msg)

            else:
                msg = self.avoid(msg)

        else:
            msg = self.stop(msg)
            self.state = "DONE"

        return msg

    def turnAngle(self, msg, angle):
        turnSpeed = 1.0 #rad/s

        time = math.radians(angle) / turnSpeed
        
        if self.elapsed < time:
            msg.linear.x = 0.0
            msg.angular.z = turnSpeed
            turn_status = False
            self.get_logger().info(f"Turning: {angle}, ")

        else:
            self.elapsed = 0.0
            turn_status = True

        return msg, turn_status

    def wait(self):
        return self.elapsed > WAIT_TIME

    def stop(self, msg):
        msg.linear.x = 0.0
        msg.angular.z = 0.0

        return msg

    def search(self, msg):

        if self.nearest_front > AVOID_DISTANCE:
            msg.linear.x = FORWARD_SPEED
            msg.angular.z = 0.0
            self.last_turn = 'FORWARD'
            self.get_logger().info(f"No obstacle: {self.nearest_front}")
            msg = self.getAwayHeading(msg)

        else:
            self.get_logger().info(f"Approaching obstacle: {self.nearest_front}")
            msg = self.avoid(msg)

        return msg

    def angleDiff(self, a1, a2):
        diff = a1 - a2

        return math.atan2(math.sin(diff), math.cos(diff))

    def find_return_heading(self, msg):
        dx = 0.0 - self.current_x
        dy = 0.0 - self.current_y

        target_angle = math.atan2(dy, dx)
        heading_err = self.angleDiff(target_angle, self.current_yaw)

        if abs(heading_err) > MAX_HEADING_ERR:
        # Rotate to face origin
            #msg.linear.x = 0.0
            msg.angular.z = TURN_SPEED if heading_err > 0 else -TURN_SPEED
            self.get_logger().info(f"Origin heading error too high, adjusting: {heading_err}")

        # else:
        # # Drive toward origin
        #     msg.linear.x = FORWARD_SPEED
        #     msg.angular.z = 0.0
        
        return msg

    def getAwayHeading(self, msg):
        dx = 0.0 - self.current_x
        dy = 0.0 - self.current_y

        target_angle = math.atan2(dy, dx)
        heading_err = self.angleDiff(target_angle, self.current_yaw)
        heading_err = math.pi - heading_err
        #heading_err = self._angle_diff(target_angle, self.current_yaw)

        if abs(heading_err) > MAX_HEADING_ERR:
        # Rotate to face origin
            #msg.linear.x = 0.0
            msg.angular.z = TURN_SPEED if heading_err > 0 else -TURN_SPEED
            self.get_logger().info(f"Heading away error too high, adjusting: {heading_err}")
        # else:
        # # Drive toward origin
        #     msg.linear.x = FORWARD_SPEED
        #     msg.angular.z = 0.0
        
        return msg

def main(args=None):
    rclpy.init(args=args)
    node = AutonomousSearch()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
