from px4_control_utils.vehicles.platform_interface import PlatformConfig
from .thrust_throttle_conversion import (
    get_throttle_command_from_force,
    get_force_from_throttle_command
)
from .constants import MASS


class GZX500Platform(PlatformConfig):
    """Platform configuration for Gazebo X500 simulation."""

    @property
    def mass(self) -> float:
        """Return the mass of the GZ X500 platform."""
        return MASS

    def get_throttle_from_force(self, force: float) -> float:
        """Convert thrust force to throttle command for GZ X500."""
        return get_throttle_command_from_force(force)

    def get_force_from_throttle(self, throttle: float) -> float:
        """Convert throttle command to thrust force for GZ X500."""
        return get_force_from_throttle_command(throttle)
