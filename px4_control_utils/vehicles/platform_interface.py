"""
Platform Interface Module
==========================

This module defines the platform abstraction layer using the Strategy Pattern and
Dependency Injection to enable seamless switching between different hardware platforms
(simulation vs. real hardware) without changing core control logic.

Design Philosophy:
-----------------
1. **Abstraction over Implementation**: Define what platforms must do, not how they do it
2. **Open/Closed Principle**: Open for extension (add new platforms), closed for modification
3. **Dependency Injection**: Core code depends on abstractions, not concrete implementations
4. **Registry Pattern**: Centralized mapping enables runtime platform selection

Benefits:
---------
- Add new platforms without modifying existing code
- Swap platforms at runtime by changing a single parameter
- Test with mock platforms easily
- Platform-specific details are encapsulated

Example Usage:
-------------
    # Get platform from registry using dependency injection
    platform_class = PLATFORM_REGISTRY[PlatformType.SIM]
    platform = platform_class()

    # Use platform interface consistently
    throttle = platform.get_throttle_from_force(20.0)
    mass = platform.mass
"""

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Dict, Type


class PlatformType(StrEnum):
    """Enumeration of supported platform types.

    Using StrEnum allows type-safe string comparison while maintaining
    readability in logs and configuration files.
    """
    SIM = "sim"
    HARDWARE = "hw"


class PlatformConfig(ABC):
    """Abstract base class defining the interface all platforms must implement.

    This class uses the Template Method pattern to define the structure that all
    platform implementations must follow. Each concrete platform (GZX500, Holybro, etc.)
    must implement these methods with platform-specific logic.

    Design Rationale:
    ----------------
    - **ABC (Abstract Base Class)**: Enforces that subclasses implement all abstract methods
    - **Property for mass**: Ensures mass is read-only and consistent across platforms
    - **Conversion methods**: Encapsulates platform-specific thrust/throttle relationships

    Why this abstraction?
    --------------------
    Different platforms have different:
    - Physical properties (mass, motor constants)
    - Thrust-to-throttle conversion equations
    - Hardware constraints

    By abstracting these differences, control algorithms can work with any platform
    without knowing implementation details.
    """

    @property
    @abstractmethod
    def mass(self) -> float:
        """Return the mass of the platform in kilograms.

        Returns:
            float: Platform mass in kg

        Note: Implemented as a property to ensure it's read-only and
              can be computed or retrieved from configuration.
        """
        pass

    @abstractmethod
    def get_throttle_from_force(self, force: float) -> float:
        """Convert thrust force to throttle command.

        Each platform has a unique relationship between desired thrust force
        and the throttle command sent to motors. This can be:
        - Linear (simple scaling)
        - Polynomial (empirically derived)
        - Physics-based (motor models)

        Args:
            force: Desired thrust force in Newtons

        Returns:
            float: Throttle command (platform-specific range, typically 0-1)

        Example:
            For a quadrotor needing 20N thrust:
            >>> throttle = platform.get_throttle_from_force(20.0)
        """
        pass

    @abstractmethod
    def get_force_from_throttle(self, throttle: float) -> float:
        """Convert throttle command to thrust force.

        Inverse of get_throttle_from_force(). Useful for:
        - Feedforward control
        - Simulation
        - Thrust estimation

        Args:
            throttle: Throttle command (platform-specific range)

        Returns:
            float: Expected thrust force in Newtons
        """
        pass


# ============================================================================
# Platform Registry - Dependency Injection
# ============================================================================

# Import concrete platform implementations
from px4_control_utils.vehicles.gz_x500.platform import GZX500Platform
from px4_control_utils.vehicles.holybro_x500V2.platform import HolybroX500V2Platform


# Registry maps platform types to their concrete implementations
PLATFORM_REGISTRY: Dict[PlatformType, Type[PlatformConfig]] = {
    PlatformType.SIM: GZX500Platform,
    PlatformType.HARDWARE: HolybroX500V2Platform
}
"""
Platform Registry - The Dependency Injection Container
=====================================================

This registry implements the Registry Pattern and enables Dependency Injection.

How it works:
1. Main node requests a platform by type (e.g., PlatformType.SIM)
2. Registry returns the appropriate class (not instance)
3. Main node instantiates the platform
4. All subsequent code uses the platform through the abstract interface

Adding a New Platform:
---------------------
1. Create new platform class inheriting from PlatformConfig
2. Implement all abstract methods
3. Add to PLATFORM_REGISTRY
4. No other code needs to change!

Example:
    class MyNewPlatform(PlatformConfig):
        @property
        def mass(self) -> float:
            return 2.5
        # ... implement other methods

    PLATFORM_REGISTRY[PlatformType.MY_NEW] = MyNewPlatform
"""
