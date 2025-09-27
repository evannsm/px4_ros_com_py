from px4_msgs.msg import(
    OffboardControlMode, VehicleCommand,
    TrajectorySetpoint, VehicleRatesSetpoint,
)


# --- Vehicle Command Publisher Functions --- #
def arm(node):
    """Send an arm command to the vehicle."""
    publish_vehicle_command(node,
        VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
    node.get_logger().info('Arm command sent')

def disarm(node):
    """Send a disarm command to the vehicle."""
    publish_vehicle_command(node,
        VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
    node.get_logger().info('Disarm command sent')

def engage_offboard_mode(node):
    """Switch to offboard mode."""
    publish_vehicle_command(node,
        VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
    node.get_logger().info("Switching to offboard mode")

def land(node):
    """Switch to land mode."""
    publish_vehicle_command(node,
        VehicleCommand.VEHICLE_CMD_NAV_LAND)
    node.get_logger().info("Switching to land mode")

def publish_vehicle_command(node, command, **params) -> None:
    """Publish a vehicle command."""
    msg = VehicleCommand()
    msg.command = command
    msg.param1 = params.get("param1", 0.0)
    msg.param2 = params.get("param2", 0.0)
    msg.param3 = params.get("param3", 0.0)
    msg.param4 = params.get("param4", 0.0)
    msg.param5 = params.get("param5", 0.0)
    msg.param6 = params.get("param6", 0.0)
    msg.param7 = params.get("param7", 0.0)
    msg.target_system = 1
    msg.target_component = 1
    msg.source_system = 1
    msg.source_component = 1
    msg.from_external = True
    msg.timestamp = int(node.get_clock().now().nanoseconds / 1000)
    node.vehicle_command_publisher.publish(msg)





# --- Offboard Control Mode Heartbeat Functions --- #
def publish_offboard_control_heartbeat_signal_position(node):
    """Publish the offboard control mode."""
    msg = OffboardControlMode()
    msg.position = True
    msg.velocity = False
    msg.acceleration = False
    msg.attitude = False
    msg.body_rate = False
    msg.timestamp = int(node.get_clock().now().nanoseconds / 1000)
    node.offboard_control_mode_publisher.publish(msg)


def publish_offboard_control_heartbeat_signal_bodyrate(node):
    """Publish the offboard control mode."""
    msg = OffboardControlMode()
    msg.position = False
    msg.velocity = False
    msg.acceleration = False
    msg.attitude = False
    msg.body_rate = True
    msg.timestamp = int(node.get_clock().now().nanoseconds / 1000)
    node.offboard_control_mode_publisher.publish(msg)





# --- Setpoint Publisher Functions --- #
def publish_position_setpoint(node, x: float, y: float, z: float, yaw: float):
    """Publish the position setpoint."""
    msg = TrajectorySetpoint()
    msg.position = [x, y, z]
    msg.yaw = yaw
    msg.timestamp = int(node.get_clock().now().nanoseconds / 1000)
    node.trajectory_setpoint_publisher.publish(msg)
    node.get_logger().info(f"Publishing position setpoints {[x, y, z, yaw]}")


def publish_rates_setpoint(node, thrust: float, roll: float, pitch: float, yaw: float) -> None:
    """Publish the thrust and body rate setpoint."""
    msg = VehicleRatesSetpoint()
    msg.roll = float(roll)
    msg.pitch = float(pitch)
    msg.yaw = float(yaw)
    msg.thrust_body[0] = 0.0
    msg.thrust_body[1] = 0.0
    msg.thrust_body[2] = -1* float(thrust)

    msg.timestamp = int(node.get_clock().now().nanoseconds / 1000)
    node.rates_setpoint_publisher.publish(msg)

    node.get_logger().info(f"Publishing rates setpoints [thrust, r,p,y]: {[thrust, roll, pitch, yaw]}")
