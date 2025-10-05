import rclpy, sys, traceback, argparse, os, inspect

from Logger import Logger  # type: ignore
from .ros2px4_node import OffboardControl

from px4_control_utils.main_utils import BANNER
from px4_control_utils.vehicles.platform_interface import PlatformType
from px4_control_utils.trajectories.trajectory_interface import TrajectoryType
from px4_control_utils.controllers.control_interface import ControllerType


# ~~ Entry point of the code -> Initializes the node and spins it. Also handles exceptions and logging ~~
def main():
    def shutdown_logging(*args):
        print("\nInterrupt/Error/Termination Detected, Triggering Logging Process and Shutting Down Node...")

        try:
            if logger:
                logger.log(offboard_control)
            offboard_control.destroy_node()
        except Exception as e:
            frame = inspect.currentframe()
            func_name = frame.f_code.co_name if frame is not None else "<unknown>"
            print(f"\nError in {__name__}:{func_name}: {e}")
            traceback.print_exc()

        # Guard shutdown so it's called at most once
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception as e:
            print(f"\nError in {__name__}: {e}")
            traceback.print_exc()

    parser = argparse.ArgumentParser()

    # -- Choose platform explicitly
    parser.add_argument(
        "--platform",
        type=PlatformType,
        choices=list(PlatformType),
        required=True,
        help="Platform type to use. Options: "
            + ", ".join(e.value for e in PlatformType)
            + ".",
    )

    # -- Controller enum
    parser.add_argument(
        "--controller",
        type=ControllerType,
        choices=list(ControllerType),
        required=True,
        help="Controller type to use. Options: "
            + ", ".join(e.value for e in ControllerType)
            + ".",
    )

    # -- Trajectory enum (parsed to TrajectoryType; keys map to TRAJ_REGISTRY via .value)
    parser.add_argument(
        "--trajectory",
        type=TrajectoryType,                 # ✔ parses "fig8_vert_tall" -> TrajectoryType.fig8_vert_tall
        choices=list(TrajectoryType),        # ✔ choices are enum members
        required=True,
        help="Trajectory type to follow. Options: "
            + ", ".join(e.value for e in TrajectoryType)
            + ". Default is 'hover'.",
    )

    parser.add_argument(
        "--double-speed",
        action="store_true",
        help="Toggle double speed for trajectory (default: False).",
    )

    parser.add_argument(
        "--spin",
        action="store_true",
        help="Toggle spin for trajectory (default: False).",
    )

    parser.add_argument(
        "--hover-mode",
        type=int,
        default=1,
        choices=range(1, 10),
        help="Mode for trajectory (hover only). Options: 1-9 (default: 1).",
    )

    parser.add_argument(
        "--short",
        action="store_false",
        help="Toggle short for trajectory (default: True).",
    )

    # -- Log file name
    parser.add_argument(
        "--log-file",
        required=True,
        help="Write a log file name. Ex: 'log.log'",
    )

    args, unknown = parser.parse_known_args(sys.argv[1:])

    # Derive booleans / values from enums
    platform: PlatformType = args.platform
    controller: ControllerType = args.controller
    trajectory: TrajectoryType = args.trajectory
    double_speed: bool = args.double_speed
    spin: bool = args.spin
    hover_mode: int = args.hover_mode
    short: bool = args.short
    filename: str = args.log_file
    base_path = os.path.dirname(os.path.abspath(__file__))

    rclpy.init()
    offboard_control = OffboardControl(platform, controller, trajectory, double_speed, spin, hover_mode, short)

    logger = None
    try:
        logger = Logger(filename, base_path)
        rclpy.spin(offboard_control)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected (Ctrl+C), exiting...")
    except Exception as e:
        frame = inspect.currentframe()
        func_name = frame.f_code.co_name if frame is not None else "<unknown>"
        print(f"\nError in {__name__}:{func_name}: {e}")
        traceback.print_exc()
    finally:
        shutdown_logging()
        print(f"{BANNER}Node has shut down.{BANNER}")

if __name__ == '__main__':
    main()
