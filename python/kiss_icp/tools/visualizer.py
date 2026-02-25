import datetime
import importlib
import os
from abc import ABC

import numpy as np

# --- BUTTONS & CONFIG ---
START_BUTTON = " START\n[SPACE]"
PAUSE_BUTTON = " PAUSE\n[SPACE]"
NEXT_FRAME_BUTTON = "NEXT FRAME\n\t\t  [N]"
SCREENSHOT_BUTTON = "SCREENSHOT\n\t\t   [S]"
LOCAL_VIEW_BUTTON = "LOCAL VIEW\n\t\t [G]"
GLOBAL_VIEW_BUTTON = "GLOBAL VIEW\n\t\t   [G]"
CENTER_VIEWPOINT_BUTTON = "CENTER VIEWPOINT\n\t\t\t\t[C]"
QUIT_BUTTON = "QUIT\n  [Q]"

# Colors (Cập nhật cho Hybrid)
BACKGROUND_COLOR = [0.0, 0.0, 0.0]
PLANAR_COLOR = [0.1176, 0.5333, 0.8980]      # Xanh dương (Planar Points)
NON_PLANAR_COLOR = [0.8470, 0.1058, 0.3764]  # Đỏ hồng (Non-Planar Points)
LOCAL_MAP_COLOR = [0.0, 0.3019, 0.2509]      # Xanh lá đậm (Map)
TRAJECTORY_COLOR = [1, 0.7568, 0.0274]       # Vàng (Quỹ đạo)

# Size constants
PTS_SIZE = 0.08
MAP_PTS_SIZE = 0.06

class StubVisualizer(ABC):
    def __init__(self): pass
    def update(self, planar_pts, non_planar_pts, target_map, pose, vis_infos): pass
    def close(self): pass


class Kissualizer(StubVisualizer):
    def __init__(self):
        try:
            self._ps = importlib.import_module("polyscope")
            self._gui = self._ps.imgui
        except ModuleNotFoundError as err:
            print(f'polyscope is not installed on your system, run "pip install polyscope"')
            exit(1)

        # Initialize GUI controls
        self._background_color = BACKGROUND_COLOR
        self._pts_size = PTS_SIZE
        self._map_size = MAP_PTS_SIZE
        
        self._block_execution = True
        self._play_mode = False
        
        # Toggles
        self._toggle_planar = True
        self._toggle_non_planar = True
        self._toggle_map = True
        self._global_view = False

        # Create data
        self._trajectory = []
        self._last_pose = np.eye(4)
        self._vis_infos = dict()
        self._selected_pose = ""

        self._initialize_visualizer()

    def update(self, planar_pts, non_planar_pts, target_map, pose, vis_infos: dict):
        self._vis_infos = dict(sorted(vis_infos.items(), key=lambda item: len(item[0])))
        # Đổi tham số nhận vào
        self._update_geometries(planar_pts, non_planar_pts, target_map, pose)
        self._last_pose = pose
        
        while self._block_execution:
            self._ps.frame_tick()
            if self._play_mode:
                break
        self._block_execution = not self._block_execution
        
    def close(self):
        self._ps.unshow()

    def _initialize_visualizer(self):
        self._ps.set_program_name("Hybrid ICP Visualizer")
        self._ps.init()
        self._ps.set_ground_plane_mode("none")
        self._ps.set_background_color(BACKGROUND_COLOR)
        self._ps.set_verbosity(0)
        self._ps.set_user_callback(self._main_gui_callback)
        self._ps.set_build_default_gui_panels(False)

    def _update_geometries(self, planar_pts, non_planar_pts, target_map, pose):
        # 1. PLANAR POINTS (BLUE)
        if planar_pts.shape[0] > 0:
            planar_cloud = self._ps.register_point_cloud(
                "planar_points", planar_pts, color=PLANAR_COLOR, point_render_mode="quad"
            )
            planar_cloud.set_radius(self._pts_size, relative=False)
            if self._global_view:
                planar_cloud.set_transform(pose)
            else:
                planar_cloud.set_transform(np.eye(4))
            planar_cloud.set_enabled(self._toggle_planar)

        # 2. NON-PLANAR POINTS (RED)
        if non_planar_pts.shape[0] > 0:
            non_planar_cloud = self._ps.register_point_cloud(
                "non_planar_points", non_planar_pts, color=NON_PLANAR_COLOR, point_render_mode="quad"
            )
            non_planar_cloud.set_radius(self._pts_size, relative=False)
            if self._global_view:
                non_planar_cloud.set_transform(pose)
            else:
                non_planar_cloud.set_transform(np.eye(4))
            non_planar_cloud.set_enabled(self._toggle_non_planar)

        # 3. LOCAL MAP (GREEN)
        if target_map.shape[0] > 0:
            map_cloud = self._ps.register_point_cloud(
                "local_map", target_map, color=LOCAL_MAP_COLOR, point_render_mode="quad",
            )
            map_cloud.set_radius(self._map_size, relative=False)
            if self._global_view:
                map_cloud.set_transform(np.eye(4))
            else:
                map_cloud.set_transform(np.linalg.inv(pose))
            map_cloud.set_enabled(self._toggle_map)

        # 4. TRAJECTORY (YELLOW)
        self._trajectory.append(pose[:3, 3])
        if self._global_view:
            self._register_trajectory()

    def _register_trajectory(self):
        if len(self._trajectory) > 0:
            trajectory_cloud = self._ps.register_point_cloud(
                "trajectory", np.asarray(self._trajectory), color=TRAJECTORY_COLOR,
            )
            trajectory_cloud.set_radius(0.3, relative=False)

    def _unregister_trajectory(self):
        self._ps.remove_point_cloud("trajectory")

    # GUI Callbacks ---------------------------------------------------------------------------
    def _start_pause_callback(self):
        button_name = PAUSE_BUTTON if self._play_mode else START_BUTTON
        if self._gui.Button(button_name) or self._gui.IsKeyPressed(self._gui.ImGuiKey_Space):
            self._play_mode = not self._play_mode

    def _next_frame_callback(self):
        if self._gui.Button(NEXT_FRAME_BUTTON) or self._gui.IsKeyPressed(self._gui.ImGuiKey_N):
            self._block_execution = False 

    def _screenshot_callback(self):
        if self._gui.Button(SCREENSHOT_BUTTON) or self._gui.IsKeyPressed(self._gui.ImGuiKey_S):
            image_filename = "hybrid_shot_" + (
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".jpg"
            )
            self._ps.screenshot(image_filename)

    def _vis_infos_callback(self):
        if self._gui.TreeNodeEx("Odometry Information", self._gui.ImGuiTreeNodeFlags_DefaultOpen):
            for key in self._vis_infos:
                self._gui.TextUnformatted(f"{key}: {self._vis_infos[key]}")
            if not self._play_mode and self._global_view:
                self._gui.TextUnformatted(f"Selected Pose: {self._selected_pose}")
            self._gui.TreePop()

    def _center_viewpoint_callback(self):
        if self._gui.Button(CENTER_VIEWPOINT_BUTTON) or self._gui.IsKeyPressed(self._gui.ImGuiKey_C):
            self._ps.reset_camera_to_home_view()

    def _toggle_buttons_andslides_callback(self):
        # Current Frame Points (Planar + Non-Planar)
        changed, self._pts_size = self._gui.SliderFloat(
            "##pts_size", self._pts_size, v_min=0.01, v_max=0.6
        )
        if changed:
            if self._ps.has_point_cloud("planar_points"):
                self._ps.get_point_cloud("planar_points").set_radius(self._pts_size, relative=False)
            if self._ps.has_point_cloud("non_planar_points"):
                self._ps.get_point_cloud("non_planar_points").set_radius(self._pts_size, relative=False)
        
        self._gui.SameLine()
        changed, self._toggle_planar = self._gui.Checkbox("Planar (Blue)", self._toggle_planar)
        if changed and self._ps.has_point_cloud("planar_points"):
            self._ps.get_point_cloud("planar_points").set_enabled(self._toggle_planar)

        self._gui.SameLine()
        changed, self._toggle_non_planar = self._gui.Checkbox("Non-Planar (Red)", self._toggle_non_planar)
        if changed and self._ps.has_point_cloud("non_planar_points"):
            self._ps.get_point_cloud("non_planar_points").set_enabled(self._toggle_non_planar)

        # LOCAL MAP
        changed, self._map_size = self._gui.SliderFloat(
            "##map_size", self._map_size, v_min=0.01, v_max=0.6
        )
        if changed and self._ps.has_point_cloud("local_map"):
            self._ps.get_point_cloud("local_map").set_radius(self._map_size, relative=False)
        self._gui.SameLine()
        changed, self._toggle_map = self._gui.Checkbox("Map (Green)", self._toggle_map)
        if changed and self._ps.has_point_cloud("local_map"):
            self._ps.get_point_cloud("local_map").set_enabled(self._toggle_map)

    def _background_color_callback(self):
        changed, self._background_color = self._gui.ColorEdit3("Background Color", self._background_color)
        if changed:
            self._ps.set_background_color(self._background_color)

    def _global_view_callback(self):
        button_name = LOCAL_VIEW_BUTTON if self._global_view else GLOBAL_VIEW_BUTTON
        if self._gui.Button(button_name) or self._gui.IsKeyPressed(self._gui.ImGuiKey_G):
            self._global_view = not self._global_view
            inv_pose = np.linalg.inv(self._last_pose)
            
            # Cập nhật hiển thị
            if self._global_view:
                if self._ps.has_point_cloud("planar_points"):
                    self._ps.get_point_cloud("planar_points").set_transform(self._last_pose)
                if self._ps.has_point_cloud("non_planar_points"):
                    self._ps.get_point_cloud("non_planar_points").set_transform(self._last_pose)
                if self._ps.has_point_cloud("local_map"):
                    self._ps.get_point_cloud("local_map").set_transform(np.eye(4))
                self._register_trajectory()
            else:
                if self._ps.has_point_cloud("planar_points"):
                    self._ps.get_point_cloud("planar_points").set_transform(np.eye(4))
                if self._ps.has_point_cloud("non_planar_points"):
                    self._ps.get_point_cloud("non_planar_points").set_transform(np.eye(4))
                if self._ps.has_point_cloud("local_map"):
                    self._ps.get_point_cloud("local_map").set_transform(inv_pose)
                self._unregister_trajectory()
                
            self._ps.reset_camera_to_home_view()

    def _quit_callback(self):
        self._gui.SetCursorPosX(self._gui.GetCursorPosX() + self._gui.GetContentRegionAvail()[0] - 50)
        if (self._gui.Button(QUIT_BUTTON) or self._gui.IsKeyPressed(self._gui.ImGuiKey_Escape) or self._gui.IsKeyPressed(self._gui.ImGuiKey_Q)):
            print("Destroying Visualizer")
            self._ps.unshow()
            os._exit(0)

    def _trajectory_pick_callback(self):
        if self._gui.GetIO().MouseClicked[0]:
            pick_selection = self._ps.get_selection()
            name = pick_selection.structure_name
            if name == "trajectory" and self._ps.has_point_cloud(name):
                try:
                    idx = pick_selection.structure_data["index"]
                    if idx < len(self._trajectory):
                        pose = self._trajectory[idx]
                        self._selected_pose = f"x: {pose[0]:7.3f}, y: {pose[1]:7.3f}, z: {pose[2]:7.3f}>"
                except:
                    self._selected_pose = ""
            else:
                self._selected_pose = ""

    def _main_gui_callback(self):
        self._start_pause_callback()
        if not self._play_mode:
            self._gui.SameLine()
            self._next_frame_callback()
        self._gui.SameLine()
        self._screenshot_callback()
        self._gui.Separator()
        self._vis_infos_callback()
        self._gui.Separator()
        self._toggle_buttons_andslides_callback()
        self._background_color_callback()
        self._global_view_callback()
        self._gui.SameLine()
        self._center_viewpoint_callback()
        self._gui.Separator()
        self._quit_callback()
        self._trajectory_pick_callback()