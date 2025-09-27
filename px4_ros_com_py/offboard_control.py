#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import(
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy
)
from px4_msgs.msg import(
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleRatesSetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleOdometry,
    VehicleStatus,
    FullState
)

import time
import math as m
import numpy as np

import jax
import jax.numpy as jnp
from jax import jacfwd
from px4_control_utils.jax_utils import jit

from scipy.spatial.transform import Rotation as R

from px4_control_utils.px4_utils.core_funcs import(
    engage_offboard_mode,
    arm,
    land,
    disarm,
    publish_position_setpoint,
    publish_rates_setpoint,
    publish_offboard_control_heartbeat_signal_position,
    publish_offboard_control_heartbeat_signal_bodyrate
)

from px4_control_utils.main_utils import BANNER
from px4_control_utils.transformations.adjust_yaw import adjust_yaw
from px4_control_utils.px4_utils.flight_phases import FlightPhase
from px4_control_utils.controllers.dev_newton_raphson.nr_standard import newton_raphson_standard as newton_raphson_flow
from px4_control_utils.controllers.dev_newton_raphson.nr_enhanced import newton_raphson_enhanced
from px4_control_utils.vehicles.gz_x500.thrust_throttle_conversion import get_throttle_command_from_force
@jit
def get_circle_ref(tT):
    w = 2 * m.pi / 20

    x = 2*jnp.sin(w*tT)
    y = 2*jnp.cos(w*tT)
    z = -5.0
    yaw = 1.57079
    return jnp.array([x, y, z, yaw])

@jit
def get_circle_ref_dot(tT):
    return jacfwd(get_circle_ref, 0)(tT)

GRAVITY: float = 9.806
class OffboardControl(Node):
    """Node for controlling a vehicle in offboard mode."""

    def __init__(self) -> None:
        super().__init__('offboard_control_takeoff_and_land')
        self.sim = True

        # ----------------------- Initialize Time-related Variables --------------------------

        self.T0 = time.time() # initial time of program
        self.program_time = 0.0 # time from start of program initialized and updated later to keep track of current time in program

        self.hover_period = 15.0
        self.flight_period = 30.0
        self.trajectory_start_time = self.hover_period
        self.trajectory_end_time = self.flight_period + self.hover_period
        self.land_time = self.flight_period + 2*(self.hover_period)


        # ----------------------- ROS2 Node Stuff --------------------------
        qos_profile = QoSProfile( # Configure QoS profile for publishing and subscribing
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ----------------------- Subscribers --------------------------
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.rates_setpoint_publisher = self.create_publisher(
            VehicleRatesSetpoint, '/fmu/in/vehicle_rates_setpoint', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)


        # ----------------------- Subscribers --------------------------
        # Mocap variables
        self.mocap_k: int = -1
        self.full_rotations: int = 0
        self.mocap_initialized: bool = False
        self.full_state_available: bool = False
        # self.vehicle_local_position_subscriber = self.create_subscription(
        #     VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile)
        # self.vehicle_odometry_subscriber = self.create_subscription(
        #     VehicleOdometry, '/fmu/out/vehicle_odometry', self.vehicle_odometry_callback, qos_profile)
        self.vehicle_full_state_subscriber = self.create_subscription( #subscribes to odometry data (position, velocity, attitude)
            FullState, '/merge_odom_localpos/full_state_relay', self.vehicle_full_state_callback, qos_profile)        
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)

        # ----------------------- Initialize Offboard Control Variables --------------------------
        self.vehicle_status = None
        self.offboard_setpoint_counter = 0
        self.takeoff_height = -5.0


        # ----------------------- Initialize Newton Flow Variables --------------------------
        self.T_LOOKAHEAD = 1.2 #lookahead time for prediction and reference tracking in NR controller
        self.LOOKAHEAD_STEP = 0.05 #step lookahead for prediction and reference tracking in NR controller
        self.INTEGRATION_STEP = 0.01 #integration step for NR controller
        self.LAST_ALPHA = jnp.array([20.0, 30.0, 30.0, 30.0]) #initial ALPHA for NR controller
        self.rng = jax.random.PRNGKey(0)

        self.ALPHA_OPTIMIZATION_METHOD = "sampling" #"sampling" # "gradient", "grid", "sampling"

        self.nr_started: bool = False # Flag to indicate if Newton-Raphson control has started
        self.nr_start_time: float = 0.0 # Time when Newton-Raphson control started
        self.trajectory_time: float = 0.0 # Time since the start of the trajectory phase

        self.MASS = 2.0
        first_thrust = self.MASS * GRAVITY # Initialize first input for hover at origin

        self._state0 = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        self._input0 = jnp.array([first_thrust, 0.1, 0.2, 0.3])
        self._ref0 = jnp.array([2.0, 2.0, -6.0, 0.0])
        self.last_input_std = self._input0
        self.last_input_enhanced = self._input0

        self.do_jit_compilation()

        self.offboard_timer_period = 0.1
        self.offboard_mode_timer = self.create_timer(self.offboard_timer_period, self.offboard_mode_timer_callback)

        self.newton_raphson_timer_period = self.INTEGRATION_STEP
        self.control_loop_timer = self.create_timer(self.newton_raphson_timer_period, self.control_loop_timer_callback)

        # self.data_log_timer_period = .1
        # self.data_log_timer = self.create_timer(self.data_log_timer_period, self.data_log_timer_callback)


    def do_jit_compilation(self) -> None:
        """ Perform a dummy call to all JIT-compiled functions to trigger compilation
            We also time them before and after JIT to ensure performance"""

        def time_fns(func):
            def wrapper(*args, **kwargs):
                time0 = time.time()
                result1 = func(*args, **kwargs)
                time1 = time.time()
                result2 = func(*args, **kwargs)
                time2 = time.time()

                tf1 = time1 - time0
                tf2 = time2 - time1
                speedup_factor = tf1 / tf2 if tf2 != 0 else 0
                print(f"\nTime taken for {func.__name__}: {time1 - time0}, Good for {1/(time1 - time0):.2f} Hz")
                print(f"Time taken for {func.__name__} (JIT): {time2 - time1}, Good for {1/(time2 - time1):.2f} Hz")
                print(f"Speedup factor for {func.__name__} (JIT): {speedup_factor}\n")

                return result2
            return wrapper
        
        # --- JIT compile NR tracker ---
        t0 = time.time()
        u, v, best_alpha, best_cost, rng = newton_raphson_flow(self._state0, self._input0, self._ref0, self.T_LOOKAHEAD, self.LOOKAHEAD_STEP, self.INTEGRATION_STEP, self.MASS, self.LAST_ALPHA, self.rng)
        u.block_until_ready()
        tf = time.time() - t0
        print(f"\nTime taken for newton_raphson_flow (first call): {tf}, Good for {1/tf:.2f} Hz")
        print(f"{u = }")

        t0 = time.time()
        u, v, best_alpha, best_cost, rng = newton_raphson_flow(self._state0, self._input0, self._ref0, self.T_LOOKAHEAD, self.LOOKAHEAD_STEP, self.INTEGRATION_STEP, self.MASS, self.LAST_ALPHA, self.rng)
        u.block_until_ready()
        tf = time.time() - t0
        print(f"\nTime taken for newton_raphson_flow (second call): {tf}, Good for {1/tf:.2f} Hz")
        print(f"{u = }")



        # Pause for 3 seconds to give myself time to read the print statements above
        print(f"Pausing for 3 seconds to read the JIT compilation times above.\n Continuing...\n")
        time.sleep(3)

    def vehicle_full_state_callback(self, msg):
        """Callback function for vehicle_full_state topic subscriber."""
        if not self.full_state_available:
            self.full_state_available = True
        self._x = msg.position[0]
        self._y = msg.position[1]
        self._z = msg.position[2] + 0.5 if (self.sim and abs(abs(msg.position[2]) - 1.0) < 2e-1) else msg.position[2]  # Adjust for sim ground level if needed

        self._vx = msg.velocity[0]
        self._vy = msg.velocity[1]
        self._vz = msg.velocity[2]

        self._ax = msg.acceleration[0]
        self._ay = msg.acceleration[1]
        self._az = msg.acceleration[2]


        self._roll, self._pitch, _yaw = R.from_quat(msg.q, scalar_first=True).as_euler('xyz', degrees=False)
        self._yaw = adjust_yaw(self, _yaw)  # Adjust yaw to account for full rotations
        self.rotation_object = R.from_euler('xyz', [self._roll, self._pitch, self._yaw], degrees=False)         # Final rotation object
        self._quaternion = self.rotation_object.as_quat()  # Quaternion representation (xyzw)


        self._angular_velocity = msg.angular_velocity

        
        self.output = np.array([self._x, self._y, self._z, self._yaw])
        self.nr_state = np.array([self._x, self._y, self._z, self._vx, self._vy, self._vz, self._roll, self._pitch, self._yaw])
        self.get_logger().info(f"Output: {[self._x, self._y, self._z, self._yaw]}")
        self.get_logger().info(f"NR State:\n{self.nr_state}")

    # def vehicle_odometry_callback(self, msg):
    #     """Callback function for vehicle_odometry topic subscriber."""
    #     self._quaternion = msg.q
    #     self._angular_velocity = msg.angular_velocity

    # def vehicle_local_position_callback(self, msg):
    #     """Callback function for vehicle_local_position topic subscriber."""

    #     self._x = msg.x
    #     self._y = msg.y
    #     self._z = msg.z

    #     self._vx = msg.vx
    #     self._vy = msg.vy
    #     self._vz = msg.vz

    #     self._ax = msg.ax
    #     self._ay = msg.ay
    #     self._az = msg.az

    def vehicle_status_callback(self, vehicle_status):
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = vehicle_status
        self.in_offboard_mode = (self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        self.armed = (self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_ARMED)
        self.in_land_mode = (self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LAND)

        if not self.in_offboard_mode:
            print(f"{BANNER}"
                  f"Not in offboard mode yet!"
                  f"Current vehicle status: {vehicle_status.nav_state}\n"
                  f"{VehicleStatus.NAVIGATION_STATE_OFFBOARD = }\n"
                  f"{self.armed=}\n"
                  f"{self.in_land_mode=}\n"
                  f"{BANNER}")
            return
        print(f"{BANNER}In Offboard Mode!{BANNER}")


    # ----------------------- Flight Phase Helpers --------------------------
    def get_phase(self) -> FlightPhase:
        t = self.program_time
        if t < self.trajectory_start_time:
            return FlightPhase.HOVER
        elif t < self.trajectory_end_time:
            return FlightPhase.CUSTOM
        elif t < self.land_time:
            return FlightPhase.RETURN
        else:
            return FlightPhase.LAND

    def time_before_next_phase(self) -> float:
        t = self.program_time
        phase = self.flight_phase
        if phase is FlightPhase.HOVER:
            return self.trajectory_start_time - t
        if phase is FlightPhase.CUSTOM:
            return self.trajectory_end_time - t
        if phase is FlightPhase.RETURN:
            return self.land_time - t
        # if phase is FlightPhase.LAND:
        #     return t - self.land_time
        return 0.0  # already in LAND
    

    def check_status_and_time(self) -> int:
        if self.vehicle_status is None: # guard to ensure vehicle status has been received
            self.get_logger().info("Waiting for vehicle status...")
            return 0          
        
        self.program_time = time.time() - self.T0  # Update program time
        self.flight_phase = self.get_phase() # Update flight phase
        self.phase_time_remaining = self.time_before_next_phase() # Update time remaining in current phase

        print(f"Program Time: {self.program_time:.2f} s")
        print(f"Remaining in {self.flight_phase.name} phase for {self.phase_time_remaining:.2f} s")
        return 1

    # ----------------------- Offboard Heartbeat Signal --------------------------
    def offboard_mode_timer_callback(self) -> None:
        """Callback function for the timer."""
        if not self.check_status_and_time(): # Ensure vehicle status is received and update time/phase before proceeding
            return
        if not self.full_state_available:
            self.get_logger().info("Waiting for full state...")
            return

        if self.offboard_setpoint_counter == 10: 
            engage_offboard_mode(self)
            arm(self)
        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter += 1

        if self.flight_phase is FlightPhase.HOVER:
            publish_offboard_control_heartbeat_signal_position(self)

        elif self.flight_phase is FlightPhase.CUSTOM:
            publish_offboard_control_heartbeat_signal_bodyrate(self)

        elif self.flight_phase is FlightPhase.RETURN:
            publish_offboard_control_heartbeat_signal_position(self)

        elif self.flight_phase is FlightPhase.LAND:
            publish_offboard_control_heartbeat_signal_position(self)

        else:
            raise ValueError("Unknown flight phase")
        
    def control_loop_timer_callback(self) -> None:
        """Callback function for the timer."""
        if not self.check_status_and_time(): # Ensure vehicle status is received and update time/phase before proceeding
            return
        if self.flight_phase is not FlightPhase.LAND and (
            not self.in_offboard_mode
            or not self.armed
            or not self.full_state_available
        ):
            self.get_logger().info("Waiting for offboard mode, arming, and full state...")
            return
        
        if self.flight_phase is FlightPhase.HOVER:
            publish_position_setpoint(self, 0.0, 0.0, self.takeoff_height, 0.0) # hover over origin at takeoff height and yaw 90 deg
        elif self.flight_phase is FlightPhase.CUSTOM:
            self.custom_control_handler()
        elif self.flight_phase is FlightPhase.RETURN:
            publish_position_setpoint(self, 2.0, -2.0, self.takeoff_height-2.0, 0.0) # hover over origin at takeoff height and yaw 90 deg
        elif self.flight_phase is FlightPhase.LAND:
            land_height = 1.0
            if abs(self._z) > land_height:
                publish_position_setpoint(self, 0.0, 0.0, -0.5, 0.0) # hover over origin at takeoff height and yaw 90 deg
            else:
                self.get_logger().info(f"Landing...")
                land(self)
            if self.in_land_mode:
                self.get_logger().info(f"Landed!")
                disarm(self)
                exit(0)
        else:
            raise ValueError("Unknown flight phase")
        

        
    def custom_control_handler(self) -> None:
        """ Custom control handler for the CUSTOM flight phase. """
        if not self.nr_started:
            self.nr_started = True
            self.nr_count = 0
            self.nr_start_time = time.time()

        self.trajectory_time = time.time() - self.nr_start_time
        self.tT = self.trajectory_time + self.T_LOOKAHEAD
        print(f"Trajectory Time: {self.trajectory_time:.2f} s, tT: {self.tT:.2f} s")

        self.ref = get_circle_ref(self.tT)
        self.ref_dot = get_circle_ref_dot(self.tT)
        print(f"{self.ref = }")
        print(f"{self.ref_dot = }")

        self.ctrl_t0 = time.time() # Control phase start time
        new_u_std, cbf_term_std, best_alpha_std, best_cost_std, rng = newton_raphson_flow(
            self.nr_state,
            self.last_input_std,
            self.ref,
            self.T_LOOKAHEAD,
            self.LOOKAHEAD_STEP,
            self.INTEGRATION_STEP,
            self.MASS,
            self.LAST_ALPHA,
            self.rng
        )
        new_u_std.block_until_ready()  # Ensure all computations are done
        
        print(f"{new_u_std = }")
        print(f"{best_alpha_std = }")
        print(f"{best_cost_std = }")
        print(f"{cbf_term_std = }")
        self.rng = rng
        self.LAST_ALPHA = best_alpha_std
        self.ctrl_tf = time.time() - self.ctrl_t0  # Control phase duration
        self.get_logger().info(f"Control computation time: {self.ctrl_tf:.4f} s, good for {1/self.ctrl_tf:.2f} Hz")

        # self.ctrl_t0 = time.time()
        # new_u_enhanced, cbf_term_enhanced, enhanced_error_term, dgdu_inv_enhanced, NR_enhanced = newton_raphson_enhanced(
        #     self.nr_state,
        #     self.last_input_enhanced,
        #     self.ref,
        #     self.T_LOOKAHEAD,
        #     self.LOOKAHEAD_STEP,
        #     self.INTEGRATION_STEP,
        #     self.MASS,
        #     self.ref_dot
        # )

        # self.ctrl_tf = time.time() - self.ctrl_t0  # Control phase duration
        # self.get_logger().info(f"Control computation time (enhanced): {self.ctrl_tf:.4f} s, good for {1/self.ctrl_tf:.2f} Hz")


        # print(f"New Input from NR (enhanced):\n{new_u_enhanced}")
        # print(f"CBF Term from NR (enhanced):\n{cbf_term_enhanced}")
        # print(f"{enhanced_error_term = }")


        new_u = new_u_std
        self.last_input_std = new_u_std
        # self.last_input_enhanced = new_u_enhanced
        new_force = new_u[0]
        new_throttle = float(get_throttle_command_from_force(new_force))
        new_roll_rate = float(new_u[1])  # Convert jax.numpy array to float
        new_pitch_rate = float(new_u[2])  # Convert jax.numpy array to float
        new_yaw_rate = float(new_u[3])    # Convert jax.numpy array to float

        publish_rates_setpoint(self, new_throttle, new_roll_rate, new_pitch_rate, new_yaw_rate)

        self.normalized_input = np.array([new_throttle, new_roll_rate, new_pitch_rate, new_yaw_rate])
        self.get_logger().info(f"New Normalized Input:\n{self.normalized_input}")
        self.get_logger().info(f"New Input:\n{new_u}")
        print(f"{BANNER}\n")

        # self.nr_count += 1
        # if self.nr_count == 10:
        #     exit(0)




def main(args=None) -> None:
    print('Starting offboard control node...')
    rclpy.init(args=args)
    offboard_control = OffboardControl()
    rclpy.spin(offboard_control)
    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)