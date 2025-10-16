# ROS 2 PX4 Offboard Control Command List

Below are organized, copy-ready command templates for running `offboard_control` with all platform and controller combinations.

---

## 🛠️ SIMULATION (Platform: `sim`)

### Regular NR
```bash
ros2 run px4_ros_com_py offboard_control --platform sim --controller nr --trajectory hover --hover-mode 1 --double-speed --log-file sim_nr_hover_m1_ds.log
ros2 run px4_ros_com_py offboard_control --platform sim --controller nr --trajectory yaw_only --double-speed --log-file sim_nr_yaw_only_ds.log
ros2 run px4_ros_com_py offboard_control --platform sim --controller nr --trajectory circle_horz --double-speed --log-file sim_nr_circle_horz_ds.log
ros2 run px4_ros_com_py offboard_control --platform sim --controller nr --trajectory circle_vert --double-speed --log-file sim_nr_circle_vert_ds.log
ros2 run px4_ros_com_py offboard_control --platform sim --controller nr --trajectory fig8_horz --double-speed --log-file sim_nr_fig8_horz_ds.log
ros2 run px4_ros_com_py offboard_control --platform sim --controller nr --trajectory fig8_vert --double-speed --short --log-file sim_nr_fig8_vert_short_ds.log
ros2 run px4_ros_com_py offboard_control --platform sim --controller nr --trajectory helix --double-speed --log-file sim_nr_helix_ds.log
ros2 run px4_ros_com_py offboard_control --platform sim --controller nr --trajectory sawtooth --double-speed --log-file sim_nr_sawtooth_ds.log
ros2 run px4_ros_com_py offboard_control --platform sim --controller nr --trajectory triangle --double-speed --log-file sim_nr_triangle_ds.log
```

### Enhanced NR
```bash
ros2 run px4_ros_com_py offboard_control --platform sim --controller nr_enhanced --trajectory hover --hover-mode 1 --double-speed --log-file sim_nr_enhanced_hover_m1_ds.log
ros2 run px4_ros_com_py offboard_control --platform sim --controller nr_enhanced --trajectory yaw_only --double-speed --log-file sim_nr_enhanced_yaw_only_ds.log
ros2 run px4_ros_com_py offboard_control --platform sim --controller nr_enhanced --trajectory circle_horz --double-speed --log-file sim_nr_enhanced_circle_horz_ds.log
ros2 run px4_ros_com_py offboard_control --platform sim --controller nr_enhanced --trajectory circle_vert --double-speed --log-file sim_nr_enhanced_circle_vert_ds.log
ros2 run px4_ros_com_py offboard_control --platform sim --controller nr_enhanced --trajectory fig8_horz --double-speed --log-file sim_nr_enhanced_fig8_horz_ds.log
ros2 run px4_ros_com_py offboard_control --platform sim --controller nr_enhanced --trajectory fig8_vert --double-speed --short --log-file sim_nr_enhanced_fig8_vert_short_ds.log
ros2 run px4_ros_com_py offboard_control --platform sim --controller nr_enhanced --trajectory helix --double-speed --log-file sim_nr_enhanced_helix_ds.log
ros2 run px4_ros_com_py offboard_control --platform sim --controller nr_enhanced --trajectory sawtooth --double-speed --log-file sim_nr_enhanced_sawtooth_ds.log
ros2 run px4_ros_com_py offboard_control --platform sim --controller nr_enhanced --trajectory triangle --double-speed --log-file sim_nr_enhanced_triangle_ds.log
```

---

## ⚙️ HARDWARE (Platform: `hw`)

### Regular NR
```bash
ros2 run px4_ros_com_py offboard_control --platform hw --controller nr --trajectory hover --hover-mode 1 --double-speed --log-file hw_nr_hover_m1_ds.log
ros2 run px4_ros_com_py offboard_control --platform hw --controller nr --trajectory yaw_only --double-speed --log-file hw_nr_yaw_only_ds.log
ros2 run px4_ros_com_py offboard_control --platform hw --controller nr --trajectory circle_horz --double-speed --log-file hw_nr_circle_horz_ds.log
ros2 run px4_ros_com_py offboard_control --platform hw --controller nr --trajectory circle_vert --double-speed --log-file hw_nr_circle_vert_ds.log
ros2 run px4_ros_com_py offboard_control --platform hw --controller nr --trajectory fig8_horz --double-speed --log-file hw_nr_fig8_horz_ds.log
ros2 run px4_ros_com_py offboard_control --platform hw --controller nr --trajectory fig8_vert --double-speed --short --log-file hw_nr_fig8_vert_short_ds.log
ros2 run px4_ros_com_py offboard_control --platform hw --controller nr --trajectory helix --double-speed --log-file hw_nr_helix_ds.log
ros2 run px4_ros_com_py offboard_control --platform hw --controller nr --trajectory sawtooth --double-speed --log-file hw_nr_sawtooth_ds.log
ros2 run px4_ros_com_py offboard_control --platform hw --controller nr --trajectory triangle --double-speed --log-file hw_nr_triangle_ds.log
```

### Enhanced NR
```bash
ros2 run px4_ros_com_py offboard_control --platform hw --controller nr_enhanced --trajectory hover --hover-mode 1 --double-speed --log-file hw_nr_enhanced_hover_m1_ds.log
ros2 run px4_ros_com_py offboard_control --platform hw --controller nr_enhanced --trajectory yaw_only --double-speed --log-file hw_nr_enhanced_yaw_only_ds.log
ros2 run px4_ros_com_py offboard_control --platform hw --controller nr_enhanced --trajectory circle_horz --double-speed --log-file hw_nr_enhanced_circle_horz_ds.log
ros2 run px4_ros_com_py offboard_control --platform hw --controller nr_enhanced --trajectory circle_vert --double-speed --log-file hw_nr_enhanced_circle_vert_ds.log
ros2 run px4_ros_com_py offboard_control --platform hw --controller nr_enhanced --trajectory fig8_horz --double-speed --log-file hw_nr_enhanced_fig8_horz_ds.log
ros2 run px4_ros_com_py offboard_control --platform hw --controller nr_enhanced --trajectory fig8_vert --double-speed --short --log-file hw_nr_enhanced_fig8_vert_short_ds.log
ros2 run px4_ros_com_py offboard_control --platform hw --controller nr_enhanced --trajectory helix --double-speed --log-file hw_nr_enhanced_helix_ds.log
ros2 run px4_ros_com_py offboard_control --platform hw --controller nr_enhanced --trajectory sawtooth --double-speed --log-file hw_nr_enhanced_sawtooth_ds.log
ros2 run px4_ros_com_py offboard_control --platform hw --controller nr_enhanced --trajectory triangle --double-speed --log-file hw_nr_enhanced_triangle_ds.log