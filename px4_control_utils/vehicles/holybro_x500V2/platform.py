from px4_control_utils.vehicles.platform_interface import PlatformConfig
from .thrust_throttle_conversion import (
    get_throttle_command_from_force,
    get_force_from_throttle_command
)
from .constants import MASS


class HolybroX500V2Platform(PlatformConfig):
    """Platform configuration for Holybro X500 V2 hardware."""

    @property
    def mass(self) -> float:
        """Return the mass of the Holybro X500 V2 platform."""
        return MASS

    def get_throttle_from_force(self, force: float) -> float:
        """Convert thrust force to throttle command for Holybro X500 V2."""
        return get_throttle_command_from_force(force)

    def get_force_from_throttle(self, throttle: float) -> float:
        """Convert throttle command to thrust force for Holybro X500 V2."""
        return get_force_from_throttle_command(throttle)
