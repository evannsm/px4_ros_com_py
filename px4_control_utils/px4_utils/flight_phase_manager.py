"""
Flight Phase Manager Module
============================

This module implements a time-based finite state machine for managing flight phases.
It encapsulates all flight phase logic and timing, separating it from the main control loop.

Design Philosophy:
-----------------
1. **State Machine Pattern**: Clean phase transitions based on time
2. **Single Responsibility**: Only manages phase timing and transitions
3. **Configuration over Code**: Phase durations are configurable
4. **Encapsulation**: Time tracking logic hidden from main node

Flight Phases:
-------------
HOVER  -> Initial stabilization phase
CUSTOM -> Execute custom trajectory with advanced control
RETURN -> Return to hover position
LAND   -> Final landing sequence

Why Time-Based?
---------------
- Predictable: Easy to plan and debug
- Safe: Guaranteed phase progression
- Simple: No complex state conditions
- Testable: Easy to simulate time progression

Benefits:
---------
- Main node doesn't manage timing directly
- Easy to modify phase durations
- Can extend to add new phases
- Clean phase transition logic
"""

from dataclasses import dataclass
from .flight_phases import FlightPhase


@dataclass
class FlightPhaseConfig:
    """Configuration for flight phase timing.

    This dataclass encapsulates all timing parameters for the flight phases,
    making it easy to adjust durations without modifying code.

    Design Rationale:
    ----------------
    - Dataclass: Immutable configuration with defaults
    - Properties: Compute transition times from base durations
    - Separate from manager: Configuration is data, manager is logic

    Timeline Example (default values):
    ---------------------------------
    t=0s  : Start, enter HOVER
    t=15s : Enter CUSTOM (trajectory_start_time)
    t=45s : Enter RETURN (trajectory_end_time)
    t=60s : Enter LAND (land_time)

    Attributes:
        hover_period: Duration of initial hover stabilization (seconds)
        flight_period: Duration of custom trajectory execution (seconds)
    """
    hover_period: float = 15.0   # Initial hover before trajectory
    flight_period: float = 30.0  # Duration of custom trajectory

    @property
    def trajectory_start_time(self) -> float:
        """Time when trajectory (CUSTOM phase) begins.

        Returns:
            float: Seconds since flight start
        """
        return self.hover_period

    @property
    def trajectory_end_time(self) -> float:
        """Time when trajectory ends and RETURN phase begins.

        Returns:
            float: Seconds since flight start
        """
        return self.flight_period + self.hover_period

    @property
    def land_time(self) -> float:
        """Time when LAND phase begins.

        Returns:
            float: Seconds since flight start

        Note: Includes another hover_period for return phase
        """
        return self.flight_period + 2 * self.hover_period


class FlightPhaseManager:
    """Manages flight phases using a time-based state machine.

    This class encapsulates all flight phase management logic, providing a clean
    interface for the main control loop to query current phase and time remaining.

    Responsibilities:
    ----------------
    1. Track elapsed flight time
    2. Determine current phase based on time
    3. Calculate time remaining in current phase
    4. Manage phase transition logic

    Design Pattern: State Machine
    -----------------------------
    - States: FlightPhase enum (HOVER, CUSTOM, RETURN, LAND)
    - Transitions: Time-based (automatic at configured intervals)
    - No actions: Phase-specific behavior handled by main node

    Why separate from main node?
    ----------------------------
    - Single Responsibility: Only manages phase logic
    - Testability: Easy to unit test phase transitions
    - Reusability: Can use same manager for different flight missions
    - Clarity: Phase logic not mixed with ROS/control code

    Example Usage:
    -------------
        >>> config = FlightPhaseConfig(hover_period=10, flight_period=20)
        >>> manager = FlightPhaseManager(config)
        >>> manager.set_start_time(time.time())
        >>>
        >>> # Later in control loop
        >>> manager.update_time(time.time())
        >>> phase = manager.get_phase()
        >>> remaining = manager.time_before_next_phase(phase)
        >>> print(f"In {phase.name}, {remaining:.1f}s remaining")
    """

    def __init__(self, config: FlightPhaseConfig = None):
        """Initialize the flight phase manager.

        Args:
            config: Flight phase configuration. If None, uses defaults from
                   FlightPhaseConfig (15s hover, 30s flight, 15s return).
        """
        self.config = config or FlightPhaseConfig()
        self._start_time: float = 0.0
        self._current_time: float = 0.0

    def set_start_time(self, time: float) -> None:
        """Set the start time of the flight.

        Should be called once at the beginning of the flight mission.

        Args:
            time: Absolute start time (e.g., time.time())
        """
        self._start_time = time

    def update_time(self, time: float) -> None:
        """Update the current time.

        Should be called at every control loop iteration to keep phase
        manager synchronized with real time.

        Args:
            time: Current absolute time (e.g., time.time())
        """
        self._current_time = time

    @property
    def program_time(self) -> float:
        """Get the elapsed time since flight start.

        Returns:
            float: Seconds elapsed since set_start_time() was called

        Note: This is relative time, not absolute time
        """
        return self._current_time - self._start_time

    def get_phase(self) -> FlightPhase:
        """Determine current flight phase based on elapsed time.

        Uses the configured phase durations to determine which phase
        the flight is currently in. Transitions happen automatically
        when time thresholds are crossed.

        Returns:
            FlightPhase: Current phase (HOVER, CUSTOM, RETURN, or LAND)

        State Transition Diagram:
        ------------------------
                    [t=0]
                      |
                      v
                 +---------+
                 | HOVER   |  Stabilize at takeoff
                 +---------+
                      |  t=trajectory_start_time
                      v
                 +---------+
                 | CUSTOM  |  Execute trajectory
                 +---------+
                      |  t=trajectory_end_time
                      v
                 +---------+
                 | RETURN  |  Return to origin
                 +---------+
                      |  t=land_time
                      v
                 +---------+
                 | LAND    |  Final landing
                 +---------+
        """
        t = self.program_time

        if t < self.config.trajectory_start_time:
            return FlightPhase.HOVER
        elif t < self.config.trajectory_end_time:
            return FlightPhase.CUSTOM
        elif t < self.config.land_time:
            return FlightPhase.RETURN
        else:
            return FlightPhase.LAND

    def time_before_next_phase(self, current_phase: FlightPhase) -> float:
        """Get time remaining before next phase transition.

        Useful for:
        - Displaying countdown to pilot
        - Pre-transition preparation
        - Logging and diagnostics

        Args:
            current_phase: The current flight phase

        Returns:
            float: Seconds until next phase begins (0.0 if in final LAND phase)

        Example:
            >>> phase = manager.get_phase()
            >>> remaining = manager.time_before_next_phase(phase)
            >>> print(f"Phase {phase.name} ends in {remaining:.1f}s")
        """
        t = self.program_time

        if current_phase is FlightPhase.HOVER:
            return self.config.trajectory_start_time - t
        elif current_phase is FlightPhase.CUSTOM:
            return self.config.trajectory_end_time - t
        elif current_phase is FlightPhase.RETURN:
            return self.config.land_time - t
        else:  # LAND
            return 0.0
