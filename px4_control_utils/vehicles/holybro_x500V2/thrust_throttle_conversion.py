import math as m


def get_throttle_command_from_force(collective_thrust: float) -> float:
    """Convert the positive collective thrust force to a positive throttle command."""
    print(f"Conv2Throttle: collective_thrust: {collective_thrust}")
    a = 0.00705385408507030
    b = 0.0807474474438391
    c = 0.0252575818743285

    # equation form is a*x + b*sqrt(x) + c = y
    throttle_command = a * collective_thrust + b * m.sqrt(collective_thrust) + c
    print(f"conv2throttle: thrust: {throttle_command = }")
    return throttle_command


def get_force_from_throttle_command(throttle_command: float) -> float:
    """Convert the positive throttle command to a positive collective thrust force."""
    print(f"Conv2Force: throttle_command: {throttle_command}")
    a = 19.2463167420814
    b = 41.8467162352942
    c = -7.19353022443441

    # equation form is a*x^2 + b*x + c = y
    collective_thrust = a * throttle_command**2 + b * throttle_command + c
    print(f"conv2force: force: {collective_thrust = }")
    return collective_thrust