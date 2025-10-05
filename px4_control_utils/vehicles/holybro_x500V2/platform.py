import math as m
from px4_control_utils.vehicles.platform_interface import PlatformConfig
from .constants import MASS


class HolybroX500V2Platform(PlatformConfig):
    """Platform configuration for Holybro X500 V2 hardware."""

    @property
    def mass(self) -> float:
        """Return the mass of the Holybro X500 V2 platform."""
        return MASS

    def get_throttle_from_force(self, force: float) -> float:
        """Convert thrust force to throttle command for Holybro X500 V2."""
        print(f"Conv2Throttle: collective_thrust: {force}")
        a = 0.00705385408507030
        b = 0.0807474474438391
        c = 0.0252575818743285

        # equation form is a*x + b*sqrt(x) + c = y
        throttle_command = a * force + b * m.sqrt(force) + c
        print(f"conv2throttle: thrust: {throttle_command = }")
        return throttle_command

    def get_force_from_throttle(self, throttle: float) -> float:
        """Convert throttle command to thrust force for Holybro X500 V2."""
        print(f"Conv2Force: throttle_command: {throttle}")
        a = 19.2463167420814
        b = 41.8467162352942
        c = -7.19353022443441

        # equation form is a*x^2 + b*x + c = y
        collective_thrust = a * throttle**2 + b * throttle + c
        print(f"conv2force: force: {collective_thrust = }")
        return collective_thrust
