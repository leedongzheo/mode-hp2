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
    # [THÊM MỚI Ở ĐÂY] Khai báo số điểm PCA
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

        # Gán giá trị đẩy xuống tầng C++
        config.adaptive_base = self.adaptive_base
        config.min_planarity_thr = self.min_planarity_thr
        config.max_planarity_thr = self.max_planarity_thr
        
        # [THÊM MỚI] Gán reg_mode xuống C++
        config.reg_mode = self.reg_mode
        # [THÊM MỚI Ở ĐÂY] Gán min_points_pca xuống C++
        config.min_points_pca = self.min_points_pca
        return config


def _to_cpp_points(frame: np.ndarray):
    """Helper: Chuyển Numpy Array sang C++ Vector3d"""
    points = np.asarray(frame, dtype=np.float64)
    return kiss_icp_pybind._Vector3dVector(points)


class KissICP:
    def __init__(self, config=None):
        # --- [THÊM ĐOẠN NÀY ĐỂ TỰ ĐỘNG NHẬN DIỆN VÀ CHUYỂN ĐỔI CONFIG CỦA SLAM] ---
        if config is not None and not hasattr(config, "_to_cpp"):
            # Nếu config truyền vào không có hàm _to_cpp (nghĩa là Pydantic config từ slam.py)
            # Ta sẽ import hàm to_kiss_config để ép kiểu nó về dạng Dataclass
            from kiss_icp.config.parser import to_kiss_config
            self.config = to_kiss_config(config)
        else:
            self.config = config or KISSConfig()
            
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

        # Nhận đúng 2 mảng điểm từ C++: preprocessed_frame (deskewed) và source (voxelized)
        deskewed_frame, source = self._pipeline._register_frame(points, ts_list)
        
        return np.asarray(deskewed_frame), np.asarray(source)
        # Lấy 2 mảng điểm từ C++
        # planar_out, non_planar_out = self._pipeline._register_frame(points, ts_list)
        
        # return np.asarray(planar_out), np.asarray(non_planar_out)

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
