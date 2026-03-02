from dataclasses import dataclass
from typing import List, Optional, Tuple, Iterable

import numpy as np

# Import thư viện pybind đã biên dịch
from kiss_icp.pybind import kiss_icp_pybind
from kiss_icp.voxelization import voxel_down_sample

@dataclass
class KISSConfig:
    voxel_size: float = 1.0
    max_range: float = 100.0
    min_range: float = 0.0
    max_points_per_voxel: int = 20
    min_motion_th: float = 0.1
    initial_threshold: float = 2.0
    max_num_iterations: int = 500
    convergence_criterion: float = 0.0001
    max_num_threads: int = 0
    deskew: bool = True
    adaptive_base: float = 0.06
    min_planarity_thr: float = 0.001
    max_planarity_thr: float = 0.2
    reg_mode: int = 0
    min_points_pca: int = 5

    def _to_cpp(self):
        """Chuyển đổi Config Python sang C++ Struct"""
        config = kiss_icp_pybind._KISSConfig()
        config.voxel_size = self.voxel_size
        config.max_range = self.max_range
        config.min_range = self.min_range
        config.max_points_per_voxel = self.max_points_per_voxel
        config.min_motion_th = self.min_motion_th
        config.initial_threshold = self.initial_threshold
        config.max_num_iterations = self.max_num_iterations
        config.convergence_criterion = self.convergence_criterion
        config.max_num_threads = self.max_num_threads
        config.deskew = self.deskew
        config.adaptive_base = self.adaptive_base
        config.min_planarity_thr = self.min_planarity_thr
        config.max_planarity_thr = self.max_planarity_thr
        config.reg_mode = self.reg_mode
        config.min_points_pca = self.min_points_pca
        return config

def _to_cpp_points(frame: np.ndarray):
    """Helper: Chuyển Numpy Array sang C++ Vector3d"""
    points = np.asarray(frame, dtype=np.float64)
    return kiss_icp_pybind._Vector3dVector(points)

# Lớp giả lập Map để lừa slam.py
class LocalMapProxy:
    def __init__(self, pipeline):
        self._pipeline = pipeline

    def point_cloud(self) -> np.ndarray:
        return np.asarray(self._pipeline._local_map())

    def clear(self):
        self._pipeline._clear_local_map()

    def add_points(self, points: np.ndarray):
        cpp_points = _to_cpp_points(points)
        self._pipeline._add_points_to_map(cpp_points)

class KissICP:
    def __init__(self, config=None):
        # Tự động nhận diện và chuyển đổi Config của SLAM
        if config is not None and not hasattr(config, "_to_cpp"):
            from kiss_icp.config.parser import to_kiss_config
            self.config = to_kiss_config(config)
        else:
            self.config = config or KISSConfig()
        
        # 1. Khởi tạo "Hộp đen" C++
        self._pipeline = kiss_icp_pybind._KissICP(self.config._to_cpp())
        
        # 2. Tạo Map Proxy để slam.py gọi các hàm .clear(), .add_points(), .point_cloud()
        self.local_map = LocalMapProxy(self._pipeline)

    def register_frame(
        self, frame: np.ndarray, timestamps: Optional[Iterable[float]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        points = _to_cpp_points(frame)
        ts_list = list(timestamps) if timestamps is not None else []
        deskewed_frame, source = self._pipeline._register_frame(points, ts_list)
        return np.asarray(deskewed_frame), np.asarray(source)

    # Getter: Đọc pose từ C++
    @property
    def last_pose(self) -> np.ndarray:
        return np.asarray(self._pipeline._last_pose())

    # Setter: Ép pose xuống C++ khi slam.py gọi "self.odometry.last_pose = np.eye(4)"
    @last_pose.setter
    def last_pose(self, pose: np.ndarray):
        self._pipeline._set_last_pose(pose)

    def reset(self):
        self._pipeline._reset()
