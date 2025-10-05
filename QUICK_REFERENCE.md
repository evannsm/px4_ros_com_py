# Quick Reference Guide

## Documentation Map

### For Understanding the System

1. **Start Here**: [`ARCHITECTURE.md`](ARCHITECTURE.md)
   - Complete overview of design philosophy
   - Architecture patterns explained
   - Data flow diagrams
   - Design decisions and rationale

2. **Module Documentation** (in-code docstrings):
   - [`platform_interface.py`](px4_control_utils/vehicles/platform_interface.py) - Platform abstraction
   - [`control_interface.py`](px4_control_utils/controllers/control_interface.py) - Controller abstraction
   - [`flight_phase_manager.py`](px4_control_utils/px4_utils/flight_phase_manager.py) - Flight phases
   - [`control_manager.py`](px4_control_utils/control_manager.py) - Control orchestration
   - [`ros2px4_node.py`](px4_ros_com_py/ros2px4_node.py) - Main ROS node

### For Extending the System

#### Add a New Platform

**Files to modify**:
1. Create: `px4_control_utils/vehicles/my_platform/platform.py`
2. Modify: `px4_control_utils/vehicles/platform_interface.py` (add to registry)

**Steps**:
```python
# 1. Create platform class
class MyPlatform(PlatformConfig):
    @property
    def mass(self) -> float:
        return 2.5

    def get_throttle_from_force(self, force: float) -> float:
        # Your conversion
        return ...

    def get_force_from_throttle(self, throttle: float) -> float:
        # Inverse conversion
        return ...

# 2. Register
PLATFORM_REGISTRY[PlatformType.MY_PLATFORM] = MyPlatform
```

See: [ARCHITECTURE.md - Adding a New Platform](ARCHITECTURE.md#adding-a-new-platform)

#### Add a New Controller

**Files to modify**:
1. Create: `px4_control_utils/controllers/my_controller.py`
2. Modify: `px4_control_utils/controllers/control_interface.py` (add to registry)

**Steps**:
```python
# 1. Create controller
class MyController:
    def compute_control(self, state, last_input, ref, ...) -> ControlOutput:
        # Your algorithm
        u = my_control_algorithm(state, ref)
        return ControlOutput(
            control_input=u,
            metadata={'diagnostic': value}
        )

# 2. Register
CTRL_REGISTRY[ControllerType.MY_CTRL] = MyController()
```

See: [ARCHITECTURE.md - Adding a New Controller](ARCHITECTURE.md#adding-a-new-controller)

#### Add a New Trajectory

**Files to modify**:
1. Create: `px4_control_utils/trajectories/my_trajectory.py`
2. Modify: `px4_control_utils/trajectories/trajectory_interface.py` (add to registry)

**Steps**:
```python
# 1. Create trajectory function
def my_trajectory(t: float, ctx: TrajContext) -> tuple:
    ref = jnp.array([x(t), y(t), z(t), yaw(t)])
    ref_dot = jnp.array([vx(t), vy(t), vz(t), yaw_rate(t)])
    return ref, ref_dot

# 2. Register
TRAJ_FUNC_REGISTRY[TrajectoryType.MY_TRAJ] = my_trajectory
```

See: [ARCHITECTURE.md - Adding a New Trajectory](ARCHITECTURE.md#adding-a-new-trajectory)

---

## Key Design Patterns

### Registry Pattern
**Where**: Platform selection, Controller selection, Trajectory selection

**Why**: Runtime component selection without if/else chains

**Example**:
```python
# Instead of conditionals:
if platform_type == "sim":
    platform = GZX500Platform()

# Use registry:
platform = PLATFORM_REGISTRY[platform_type]()
```

### Dependency Injection
**Where**: Main node initialization

**Why**: Testability, flexibility, explicit dependencies

**Example**:
```python
# Components receive dependencies:
control_manager = ControlManager(
    platform=platform,          # Injected
    controller_type=ctrl_type,  # Injected
    trajectory_type=traj_type   # Injected
)
```

### Facade Pattern
**Where**: ControlManager

**Why**: Simplify complex subsystem (controller + trajectory + platform)

**Example**:
```python
# Without facade (complex):
ref = traj_func(time, ctx)
output = controller.compute(state, ref, ...)
update_state(output.metadata)

# With facade (simple):
ref = control_manager.compute_reference(time)
output = control_manager.compute_control(state, ref)
# State managed internally
```

---

## System Flow Overview

### Initialization
```
1. Main Node creates:
   ├─→ Platform (from PLATFORM_REGISTRY)
   ├─→ FlightPhaseManager (with timing config)
   └─→ ControlManager
       ├─→ Selects controller (from CTRL_REGISTRY)
       └─→ Selects trajectory (from TRAJ_FUNC_REGISTRY)
```

### Control Loop (Every Iteration)
```
1. Update time → FlightPhaseManager
2. Get current phase
3. Based on phase:

   HOVER/RETURN:
   ├─→ Get hover reference from trajectory registry
   └─→ Publish position setpoint

   CUSTOM:
   ├─→ Compute reference (via ControlManager)
   ├─→ Compute control (via ControlManager)
   ├─→ Convert force→throttle (via Platform)
   └─→ Publish rates setpoint

   LAND:
   └─→ Execute landing sequence
```

---

## Component Responsibilities

### OffboardControl (Main Node)
- ROS2 communication (publish/subscribe)
- Flight phase coordination
- High-level control flow
- **Does NOT**: Handle platform details, control computation, phase timing

### FlightPhaseManager
- Track elapsed time
- Determine current phase
- Calculate time to next phase
- **Does NOT**: Execute phase-specific logic (that's main node's job)

### ControlManager
- Select controller and trajectory
- Manage control state (last_input, alpha, rng)
- Coordinate reference generation and control computation
- **Does NOT**: Know about ROS, flight phases, or PX4 communication

### PlatformConfig
- Provide platform mass
- Convert thrust ↔ throttle
- **Does NOT**: Know about control algorithms or trajectories

---

## Code Navigation Tips

### Finding Platform Logic
```
px4_control_utils/vehicles/
├── platform_interface.py    # Abstract interface + registry
├── gz_x500/
│   └── platform.py          # Simulation platform
└── holybro_x500V2/
    └── platform.py          # Hardware platform
```

### Finding Controller Logic
```
px4_control_utils/controllers/
├── control_interface.py     # Abstract interface + registry
├── controller_wrappers.py   # Adapters for existing controllers
└── dev_newton_raphson/      # Actual controller implementations
```

### Finding Trajectory Logic
```
px4_control_utils/trajectories/
├── trajectory_interface.py       # Registry + types
├── differentiable_trajectories.py # Main trajectories
└── trajectory_context.py         # Configuration object
```

### Finding Flight Phase Logic
```
px4_control_utils/px4_utils/
├── flight_phases.py         # Phase enum
└── flight_phase_manager.py  # Time-based state machine
```

---

## Common Questions

### Q: How do I change the trajectory?
**A**: Pass different `trajectory_type` to main node initialization. It uses the registry automatically.

### Q: How do I add a new platform?
**A**: Create class inheriting `PlatformConfig`, register in `PLATFORM_REGISTRY`. Done.

### Q: Where is the control loop?
**A**: `ros2px4_node.py::control_loop_timer_callback()` orchestrates everything.

### Q: How does the system switch controllers?
**A**: Via `CTRL_REGISTRY[controller_type]` - just pass different type at startup.

### Q: Where are phase durations configured?
**A**: `FlightPhaseConfig` in main node `__init__`. Default: 15s hover, 30s flight, 15s return.

### Q: How does JIT compilation work?
**A**: `do_jit_compilation()` calls controller once to trigger JAX compilation before flight.

### Q: What's the difference between ControlManager and ControllerInterface?
**A**:
- `ControllerInterface`: Interface for individual controllers (compute_control)
- `ControlManager`: Facade that orchestrates controller + trajectory + platform

---

## Troubleshooting

### "Platform not found in registry"
→ Check `platform_type` value matches key in `PLATFORM_REGISTRY`

### "Controller not found in registry"
→ Check `controller_type` value matches key in `CTRL_REGISTRY`

### "Trajectory not found in registry"
→ Check `trajectory_type` value matches key in `TRAJ_FUNC_REGISTRY`

### "Control computation too slow"
→ Check JIT compilation happened (`do_jit_compilation()` output)

### "Phase transitions at wrong times"
→ Modify `FlightPhaseConfig` parameters in main node `__init__`

---

## Best Practices

1. **Adding Components**: Always use registries, never modify existing code
2. **Testing**: Use dependency injection to inject mocks
3. **Debugging**: Check phase first, then control computation
4. **Performance**: Ensure JIT compilation happens before flight
5. **Documentation**: Update this guide when adding new component types

---

## Further Reading

- **Design Philosophy**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **In-Code Docs**: Check module docstrings (comprehensive explanations)
- **Patterns Reference**: Gang of Four Design Patterns book
- **JAX**: https://jax.readthedocs.io/ (for control computation)
- **ROS2**: https://docs.ros.org/en/humble/ (for node structure)
