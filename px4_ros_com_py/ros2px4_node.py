#!/usr/bin/env python3
"""
Main ROS2 Node for PX4 Offboard Control
========================================

This module implements the main ROS2 node for offboard control of PX4-based multirotor UAVs.
It orchestrates flight phases, control computation, and communication with PX4 autopilot.

Architecture Overview:
---------------------
This node follows a modular, interface-based design with clear separation of concerns:

1. **Platform Abstraction** (via PlatformConfig):
   - Handles platform-specific details (sim vs. hardware)
   - Abstracts mass, thrust conversion
   - Selected from PLATFORM_REGISTRY at runtime

2. **Flight Phase Management** (via FlightPhaseManager):
   - Time-based state machine: HOVER → CUSTOM → RETURN → LAND
   - Encapsulates phase timing and transition logic
   - Separates phase management from control logic

3. **Control Orchestration** (via ControlManager):
   - Facade for controller + trajectory + platform
   - Manages control state (last inputs, optimization parameters)
   - Provides simple interface to complex control subsystem

Design Benefits:
---------------
- **Modularity**: Swap platforms/controllers/trajectories without changing this code
- **Testability**: All dependencies injected, easy to mock
- **Clarity**: Each responsibility in separate manager
- **Extensibility**: Add new components via registries

Flow Summary:
------------
1. Initialize: Create platform, phase manager, control manager
2. Each iteration:
   a. Update flight phase (via phase_manager)
   b. Based on phase:
      - HOVER/RETURN: Use simple position control
      - CUSTOM: Use advanced control (via control_manager)
      - LAND: Execute landing sequence
   c. Publish commands to PX4

Key Dependencies:
----------------
- rclpy: ROS2 Python client library
- px4_msgs: PX4 ROS2 message types
- jax: For JIT-compiled control computations
- numpy: Numerical operations

For architectural details, see ARCHITECTURE.md
"""

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
    VehicleStatus,
    FullState,
    Logging
)

import time
import math as m
import numpy as np

import jax.numpy as jnp
from jax.random import PRNGKey
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


from px4_control_utils.vehicles.platform_interface import (
    PlatformType,
    PlatformConfig,
    PLATFORM_REGISTRY
)
from px4_control_utils.controllers.control_interface import ControllerType
from px4_control_utils.trajectories.trajectory_interface import (
    TrajectoryType,
    TRAJ_FUNC_REGISTRY
)
from px4_control_utils.trajectories import TrajContext
from px4_control_utils.px4_utils.flight_phase_manager import (
    FlightPhaseManager,
    FlightPhaseConfig
)
from px4_control_utils.control_manager import ControlManager, ControlParams

GRAVITY: float = 9.806

# ctx_hover = TrajContext(mode=6, sim=True, spin=False, double_speed=False)
# ctx_traj = TrajContext(sim=True, spin=True, double_speed=False)

class OffboardControl(Node):
    """ROS2 node for offboard control of PX4-based multirotor UAVs.

    This node implements a complete flight mission with phase-based control:
    - HOVER: Initial stabilization
    - CUSTOM: Advanced trajectory tracking with model-based control
    - RETURN: Return to origin
    - LAND: Final landing sequence

    The node uses a modular architecture with:
    - Platform abstraction (via PLATFORM_REGISTRY)
    - Flight phase management (via FlightPhaseManager)
    - Control orchestration (via ControlManager)

    All platform-specific, controller-specific, and trajectory-specific
    logic is abstracted behind clean interfaces, making this node's
    responsibilities focused on:
    1. ROS2 communication (publish/subscribe)
    2. Flight phase coordination
    3. High-level control flow

    See ARCHITECTURE.md for detailed design documentation.
    """
    def __init__(self, platform_type: PlatformType, controller_type: ControllerType, trajectory_type: TrajectoryType, double_speed: bool, spin: bool, hover_mode: int, short: bool) -> None:
        super().__init__('offboard_control_node')
        self.get_logger().info(f"{BANNER}Initializing ROS 2 node: '{self.__class__.__name__}'{BANNER}")
        self.platform_type = platform_type
        self.controller_type = controller_type
        self.trajectory_type = trajectory_type
        self.double_speed = double_speed
        self.spin = spin
        self.hover_mode = hover_mode
        self.short = short
        self.sim = platform_type==PlatformType.SIM

        print(f"{BANNER}Platform: {self.platform_type}, Controller: {self.controller_type}, Trajectory: {self.trajectory_type}, Double Speed: {self.double_speed}, Spin: {self.spin}, Hover Mode: {self.hover_mode}, Short: {self.short}{BANNER}")
        #now print the types
        print(f"{BANNER}PlatformType: {type(self.platform_type)}, ControllerType: {type(self.controller_type)}, TrajectoryType: {type(self.trajectory_type)}, Double Speed: {type(self.double_speed)}, Spin: {type(self.spin)}, Hover Mode: {type(self.hover_mode)}, Short: {type(self.short)}{BANNER}")
        print(f"str(self.platform_type) = {str(self.platform_type)}")
        # exit(0)

        # Initialize platform configuration using dependency injection
        platform_class = PLATFORM_REGISTRY[self.platform_type]
        self.platform: PlatformConfig = platform_class()

        mode_name = "SIMULATION" if self.platform_type is PlatformType.SIM else "HARDWARE"
        self.get_logger().info(f"{BANNER}Running in {mode_name} mode{BANNER}")
        self.get_logger().info(f"{BANNER}Platform mass: {self.platform.mass} kg{BANNER}")


        self.ctx_hover = TrajContext(sim=self.sim, hover_mode=self.hover_mode)
        self.ctx_traj = TrajContext(sim=self.sim, spin=self.spin, double_speed=self.double_speed, short=self.short)

        # ----------------------- Initialize Flight Phase Manager --------------------------
        flight_config = FlightPhaseConfig(hover_period=15.0, flight_period=30.0)
        self.phase_manager = FlightPhaseManager(config=flight_config)
        self.T0 = time.time()  # Initial time of program
        self.phase_manager.set_start_time(self.T0)


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
        self.log_publisher = self.create_publisher(
            Logging, '/plotjuggler/logging', qos_profile)


        # ----------------------- Subscribers --------------------------
        # Mocap variables
        self.mocap_k: int = -1
        self.full_rotations: int = 0
        self.mocap_initialized: bool = False
        self.full_state_available: bool = False
        self.vehicle_full_state_subscriber = self.create_subscription( #subscribes to odometry data (position, velocity, attitude)
            FullState, '/merge_odom_localpos/full_state_relay', self.vehicle_full_state_callback, qos_profile)        
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)

        # ----------------------- Initialize Offboard Control Variables --------------------------
        self.vehicle_status = None
        self.offboard_setpoint_counter = 0
        # self.takeoff_height = -5.0 if self.sim else -1.2
        # self.return_height = -3.0 if self.sim else -1.2


        # ----------------------- Initialize Control Manager --------------------------
        ctrl_params = ControlParams(
            t_lookahead=1.2,
            lookahead_step=0.05,
            integration_step=0.01,
            gravity=GRAVITY
        )
        self.control_manager = ControlManager(
            controller_type=self.controller_type,
            trajectory_type=self.trajectory_type,
            platform=self.platform,
            traj_context=self.ctx_traj,
            params=ctrl_params
        )

        # Trajectory tracking
        self.trajectory_started: bool = False
        self.trajectory_time: float = 0.0

        # JIT compilation test variables
        first_thrust = self.platform.mass * GRAVITY
        self._state0 = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        self._input0 = jnp.array([first_thrust, 0.1, 0.2, 0.3])
        self._ref0 = jnp.array([2.0, 2.0, -6.0, 0.0])

        self.do_jit_compilation()

        self.offboard_timer_period = 0.1
        self.offboard_mode_timer = self.create_timer(self.offboard_timer_period, self.offboard_mode_timer_callback)

        self.newton_raphson_timer_period = ctrl_params.integration_step
        self.control_loop_timer = self.create_timer(self.newton_raphson_timer_period, self.control_loop_timer_callback)

        self.data_log_timer_period = .1
        self.first_log = True
        if self.first_log:
            self.first_log = False
            msg = Logging()
            # msg.timestamp = self.get_clock().now().nanoseconds // 1000  # micro
            self.log_publisher.publish(msg)
        self.data_log_timer = self.create_timer(self.data_log_timer_period, self.data_log_timer_callback)


    def do_jit_compilation(self) -> None:
        """ Perform a dummy call to all JIT-compiled functions to trigger compilation
            We also time them before and after JIT to ensure performance"""

        # --- JIT compile controller ---
        print("\n=== JIT Compilation Test ===")
        t0 = time.time()
        output = self.control_manager.compute_control(
            state=self._state0,
            ref=self._ref0,
            ref_dot=None
        )
        output.control_input.block_until_ready()
        tf = time.time() - t0
        print(f"Time taken for controller (first call): {tf:.4f}s, Good for {1/tf:.2f} Hz")
        print(f"Control input: {output.control_input}")

        t0 = time.time()
        output = self.control_manager.compute_control(
            state=self._state0,
            ref=self._ref0,
            ref_dot=None
        )
        output.control_input.block_until_ready()
        tf2 = time.time() - t0
        print(f"Time taken for controller (second call, JIT): {tf2:.4f}s, Good for {1/tf2:.2f} Hz, Speedup: {tf/tf2:.2f}x")
        print(f"Control input: {output.control_input}")


        # Pause for 3 seconds to give myself time to read the print statements above
        print(f"\nPausing for 3 seconds to read the JIT compilation times above.\nContinuing...\n")
        time.sleep(3)

    def data_log_timer_callback(self) -> None:
        if self.flight_phase is not FlightPhase.CUSTOM:
            return
        msg = Logging()

        msg.timestamp = self.get_clock().now().nanoseconds // 1000  # microseconds
        msg.traj_time = self.trajectory_time

        msg.platform = str(self.platform_type)
        msg.controller = str(self.controller_type)
        msg.trajectory = str(self.trajectory_type)
        msg.traj_double = bool(self.double_speed)
        msg.traj_spin = bool(self.spin)
        msg.traj_short = bool(self.short)

        # --- Pose and Derivatives ---
        msg.x = float(self._x)
        msg.y = float(self._y)
        msg.z = float(self._z)

        msg.vx = float(self._vx)
        msg.vy = float(self._vy)
        msg.vz = float(self._vz)

        msg.ax = float(self._ax)
        msg.ay = float(self._ay)
        msg.az = float(self._az)

        msg.roll = float(self._roll)
        msg.pitch = float(self._pitch)
        msg.yaw = float(self._yaw)

        msg.p = float(self._angular_velocity[0])
        msg.q = float(self._angular_velocity[1])
        msg.r = float(self._angular_velocity[2])

        # --- Reference and Derivatives ---
        msg.x_ref = float(self.ref[0])
        msg.y_ref = float(self.ref[1])
        msg.z_ref = float(self.ref[2])

        msg.vx_ref = float(self.ref_dot[0])
        msg.vy_ref = float(self.ref_dot[1])
        msg.vz_ref = float(self.ref_dot[2])

        msg.yaw_ref = float(self.ref[3])
        msg.yawd_ref = float(self.ref_dot[3])

        # --- Control Inputs ---
        msg.u_throttle = float(self.normalized_input[0])
        msg.u_p = float(self.normalized_input[1])
        msg.u_q = float(self.normalized_input[2])
        msg.u_r = float(self.normalized_input[3])

        # --- Publish ---
        self.log_publisher.publish(msg)

    def vehicle_full_state_callback(self, msg):
        """Callback function for vehicle_full_state topic subscriber."""
        if not self.full_state_available:
            self.full_state_available = True
        self._x = msg.position[0]
        self._y = msg.position[1]
        self._z = (msg.position[2] + 0.5) if (self.sim and (abs(msg.position[2]) < 1.2)) else msg.position[2]  # Adjust for sim ground level if needed

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
    def check_status_and_time(self) -> int:
        if self.vehicle_status is None: # guard to ensure vehicle status has been received
            self.get_logger().info("Waiting for vehicle status...")
            return 0

        # Update phase manager time
        self.phase_manager.update_time(time.time())

        # Update flight phase
        self.flight_phase = self.phase_manager.get_phase()
        self.phase_time_remaining = self.phase_manager.time_before_next_phase(self.flight_phase)

        print(f"Program Time: {self.phase_manager.program_time:.2f} s")
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
            # Use trajectory registry to get hover trajectory
            hover_traj = TRAJ_FUNC_REGISTRY[TrajectoryType.HOVER]
            ref, _ = np.array(hover_traj(self.phase_manager.program_time, self.ctx_hover))
            print(f"Hover Phase: {ref=}")
            publish_position_setpoint(self, ref[0], ref[1], ref[2], ref[3])
            
        elif self.flight_phase is FlightPhase.CUSTOM:
            self.custom_control_handler()

        elif self.flight_phase is FlightPhase.RETURN:
            # Use trajectory registry to get hover trajectory
            hover_traj = TRAJ_FUNC_REGISTRY[TrajectoryType.HOVER]
            ref, _ = np.array(hover_traj(self.phase_manager.program_time, self.ctx_hover))
            print(f"Return Phase: {ref=}")
            publish_position_setpoint(self, ref[0], ref[1], ref[2], ref[3])
            
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
        if not self.trajectory_started:
            self.trajectory_started = True
            self.control_manager.start_trajectory(time.time())

        # Compute trajectory reference with lookahead
        self.trajectory_time = time.time() - self.control_manager._start_time
        lookahead_time = self.trajectory_time + self.control_manager.params.t_lookahead
        print(f"Trajectory Time: {self.trajectory_time:.2f} s, Lookahead: {lookahead_time:.2f} s")

        # Get reference from control manager
        self.ref, self.ref_dot = self.control_manager.compute_reference(lookahead_time)
        print(f"{self.ref = }")
        print(f"{self.ref_dot = }")

        # Compute control using control manager
        ctrl_t0 = time.time()
        output = self.control_manager.compute_control(
            state=self.nr_state,
            ref=self.ref,
            ref_dot=self.ref_dot
        )
        output.control_input.block_until_ready()  # Ensure all computations are done
        ctrl_tf = time.time() - ctrl_t0

        # Log control output
        print(f"Control input: {output.control_input}")
        print(f"Metadata: {output.metadata}")
        self.get_logger().info(f"Control computation time: {ctrl_tf:.4f} s, good for {1/ctrl_tf:.2f} Hz")

        # Convert control input to throttle and rates
        new_force = output.control_input[0]
        new_throttle = float(self.platform.get_throttle_from_force(float(new_force)))
        new_roll_rate = float(output.control_input[1])
        new_pitch_rate = float(output.control_input[2])
        new_yaw_rate = float(output.control_input[3])

        # Publish control setpoint
        publish_rates_setpoint(self, new_throttle, new_roll_rate, new_pitch_rate, new_yaw_rate)

        # Store for logging
        self.normalized_input = np.array([new_throttle, new_roll_rate, new_pitch_rate, new_yaw_rate])
        self.get_logger().info(f"New Normalized Input:\n{self.normalized_input}")
        self.get_logger().info(f"New Input:\n{output.control_input}")
        print(f"{BANNER}\n")




# def main(args=None) -> None:
#     print('Starting offboard control node...')
#     rclpy.init(args=args)
#     offboard_control = OffboardControl()
#     rclpy.spin(offboard_control)
#     offboard_control.destroy_node()
#     rclpy.shutdown()


# if __name__ == '__main__':
#     try:
#         main()
#     except Exception as e:
#         print(e)