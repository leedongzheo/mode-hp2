from typing import Optional

from pydantic import BaseModel


class DataConfig(BaseModel):
    max_range: float = 100.0
    min_range: float = 0.0
    # min_range: float = 0.5
    deskew: bool = True


class MappingConfig(BaseModel):
    voxel_size: Optional[float] = None  # default: take it from data
    max_points_per_voxel: int = 20
    # max_points_per_voxel: int = 1

class RegistrationConfig(BaseModel):
    max_num_iterations: Optional[int] = 500
    # max_num_iterations: Optional[int] = 150
    convergence_criterion: Optional[float] = 0.0001
    max_num_threads: Optional[int] = 0  # 0 means automatic
    # [THÊM MỚI] Biến cấu hình chế độ ICP
    # 0: Hybrid, 1: Point-to-Point, 2: Point-to-Plane
    reg_mode: int = 0
    # [THÊM MỚI] Số điểm tối thiểu để chạy PCA
    min_points_pca: int = 20


class AdaptiveThresholdConfig(BaseModel):
    fixed_threshold: Optional[float] = None
    initial_threshold: float = 2.0
    min_motion_th: float = 0.1
    adaptive_base: float = 0.12
    min_planarity_thr: float = 0.001
    max_planarity_thr: float = 0.2
