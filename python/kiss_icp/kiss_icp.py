from dataclasses import dataclass
from typing import List, Optional, Tuple, Iterable

import numpy as np

# Import thư viện pybind đã biên dịch
from kiss_icp.pybind import kiss_icp_pybind


@dataclass
class KISSConfig:
    # Các tham số mặc định lấy từ KISSConfig struct cũ
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

    # 3 tham số khai báo cho Adaptive Threshold
    adaptive_base: float = 0.06
    min_planarity_thr: float = 0.001
    max_planarity_thr: float = 0.2
    
    # [THÊM MỚI] Khai báo biến reg_mode (0: Hybrid, 1: Pt2Pt, 2: Pt2Plane)
    reg_mode: int = 0
    
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

        # Gán giá trị đẩy xuống tầng C++
        config.adaptive_base = self.adaptive_base
        config.min_planarity_thr = self.min_planarity_thr
        config.max_planarity_thr = self.max_planarity_thr
        
        # [THÊM MỚI] Gán reg_mode xuống C++
        config.reg_mode = self.reg_mode
        
        return config


def _to_cpp_points(frame: np.ndarray):
    """Helper: Chuyển Numpy Array sang C++ Vector3d"""
    points = np.asarray(frame, dtype=np.float64)
    return kiss_icp_pybind._Vector3dVector(points)


class KissICP:
    def __init__(self, config: Optional[KISSConfig] = None):
        # 1. Quản lý Config
        self.config = config or KISSConfig()
        
        # 2. Khởi tạo "Hộp đen" C++ (Monolithic Wrapper)
        # Toàn bộ logic Preprocess -> Voxelize -> Register -> Map Update nằm trong này
        self._pipeline = kiss_icp_pybind._KissICP(self.config._to_cpp())

    def register_frame(
        self, frame: np.ndarray, timestamps: Optional[Iterable[float]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Input: Raw Point Cloud
        Output: Tuple (Planar Points, Non-Planar Points)
        """
        points = _to_cpp_points(frame)
        ts_list = list(timestamps) if timestamps is not None else []

        # Lấy 2 mảng điểm từ C++
        planar_out, non_planar_out = self._pipeline._register_frame(points, ts_list)
        
        return np.asarray(planar_out), np.asarray(non_planar_out)

    @property
    def last_pose(self) -> np.ndarray:
        """Trả về pose cuối cùng (4x4 Matrix) từ C++"""
        return np.asarray(self._pipeline._last_pose())

    @property
    def local_map(self) -> np.ndarray:
        """Lấy toàn bộ điểm trong bản đồ hiện tại"""
        return np.asarray(self._pipeline._local_map())

    def reset(self):
        """Reset thuật toán về trạng thái ban đầu"""
        self._pipeline._reset()

# Hàm tiện ích (Global function) giống GenZ
def voxel_down_sample(frame: np.ndarray, voxel_size: float) -> np.ndarray:
    return np.asarray(kiss_icp_pybind._voxel_down_sample(_to_cpp_points(frame), voxel_size))