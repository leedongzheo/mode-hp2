import contextlib
import datetime
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
from pyquaternion import Quaternion

# Import wrapper
from kiss_icp.kiss_icp import KissICP
# Import thêm to_kiss_config
from kiss_icp.config import load_config, write_config, to_kiss_config 
from kiss_icp.config.parser import KISSConfig # [THÊM MỚI] Import
from kiss_icp.metrics import absolute_trajectory_error, sequence_error
from kiss_icp.tools.pipeline_results import PipelineResults
from kiss_icp.tools.progress_bar import get_progress_bar
from kiss_icp.tools.visualizer import Kissualizer, StubVisualizer


class OdometryPipeline:
    def __init__(
        self,
        dataset,
        config: Optional[Path] = None,
        visualize: bool = False,
        n_scans: int = -1,
        jump: int = 0,
    ):
        self._dataset = dataset
        self._n_scans = (
            len(self._dataset) - jump if n_scans == -1 else min(len(self._dataset) - jump, n_scans)
        )
        self._jump = jump
        self._first = jump
        self._last = self._jump + self._n_scans

        # Config and output dir
        # --- [SỬA LỖI KIỂU DỮ LIỆU Ở ĐÂY] ---
        if isinstance(config, KISSConfig):
            self.config = config
        else:
            self.config = load_config(config)
        # ------------------------------------
        self.results_dir = None

        # Pipeline
        # Chuyển đổi Pydantic Config -> Runtime Dataclass trước khi đưa vào Wrapper
        self.odometry = KissICP(config=to_kiss_config(self.config))
        
        self.results = PipelineResults()
        self.times = np.zeros(self._n_scans)
        self.poses = np.zeros((self._n_scans, 4, 4))
        self.has_gt = hasattr(self._dataset, "gt_poses")
        self.gt_poses = self._dataset.gt_poses[self._first : self._last] if self.has_gt else None
        self.dataset_name = self._dataset.__class__.__name__
        self.dataset_sequence = (
            self._dataset.sequence_id
            if hasattr(self._dataset, "sequence_id")
            else os.path.basename(self._dataset.data_dir)
        )

        # Visualizer
        self.visualizer = Kissualizer() if visualize else StubVisualizer()
        
        # Vis Info
        self._vis_infos = {
            "max_range": self.config.data.max_range,
            "min_range": self.config.data.min_range,
        }
        if hasattr(self._dataset, "use_global_visualizer"):
            self.visualizer._global_view = self._dataset.use_global_visualizer

    # Public interface  ------
    def run(self):
        # [THÊM MỚI] In log ra màn hình để biết đang chạy Mode nào
        mode_str = ["Hybrid (Dynamic Alpha)", "Point-to-Point (Alpha=0)", "Point-to-Plane (Alpha=1)"]
        idx_mode = self.config.registration.reg_mode
        print(f"\n[INFO] Starting {self.dataset_name} Sequence {self.dataset_sequence}")
        print(f"[INFO] Registration Mode: [{idx_mode}] {mode_str[idx_mode]}")
        print(f"[INFO] Adaptive Params: Base={self.config.adaptive_threshold.adaptive_base}, " 
              f"Min={self.config.adaptive_threshold.min_planarity_thr}, "
              f"Max={self.config.adaptive_threshold.max_planarity_thr}\n")
        
        self._run_pipeline()
        self._run_evaluation()
        self._create_output_dir()
        self._write_result_poses()
        self._write_gt_poses()
        self._write_cfg()
        self._write_log()
        self._write_graph()
        self._write_trajectory_plot()
        self._write_local_maps()
        return self.results

    # Private interface  ------
    def _run_pipeline(self):
        for idx in get_progress_bar(self._first, self._last):
            raw_frame, timestamps = self._dataset[idx]
            start_time = time.perf_counter_ns()
            
            # --- [SỬA LỖI] CHỈ GỌI REGISTER_FRAME 1 LẦN DUY NHẤT ---
            planar_pts, non_planar_pts = self.odometry.register_frame(raw_frame, timestamps)
            # --------------------------------------------------------

            self.poses[idx - self._first] = self.odometry.last_pose
            self.times[idx - self._first] = time.perf_counter_ns() - start_time

            # Update visualizer info
            self._vis_infos["FPS"] = int(np.floor(self._get_fps()))
            self._vis_infos["Planar Pts"] = planar_pts.shape[0]     
            self._vis_infos["Non-Planar"] = non_planar_pts.shape[0] 

            self.visualizer.update(
                planar_pts,        # Truyền mảng Planar
                non_planar_pts,    # Truyền mảng Non-Planar
                self.odometry.local_map,
                self.odometry.last_pose,
                self._vis_infos,
            )

        self.visualizer.close()

    @staticmethod
    def save_poses_kitti_format(filename: str, poses: np.ndarray):
        np.savetxt(fname=f"{filename}_kitti.txt", X=poses[:, :3].reshape(-1, 12))

    @staticmethod
    def save_poses_tum_format(filename, poses, timestamps):
        def _to_tum_format(_poses, _timestamps):
            tum_data = np.zeros((len(_poses), 8))
            with contextlib.suppress(ValueError):
                for idx in range(len(_poses)):
                    tx, ty, tz = _poses[idx, :3, -1].flatten()
                    qw, qx, qy, qz = Quaternion(matrix=_poses[idx], atol=0.01).elements
                    tum_data[idx] = np.r_[float(_timestamps[idx]), tx, ty, tz, qx, qy, qz, qw]
            return tum_data.astype(np.float64)

        np.savetxt(fname=f"{filename}_tum.txt", X=_to_tum_format(poses, timestamps), fmt="%.4f")

    def _calibrate_poses(self, poses):
        return (
            self._dataset.apply_calibration(poses)
            if hasattr(self._dataset, "apply_calibration")
            else poses
        )

    def _get_frames_timestamps(self):
        return (
            self._dataset.get_frames_timestamps()
            if hasattr(self._dataset, "get_frames_timestamps")
            else np.arange(0, self._n_scans, 1.0)
        )

    def _save_poses(self, filename: str, poses, timestamps):
        np.save(filename, poses)
        self.save_poses_kitti_format(filename, poses)
        self.save_poses_tum_format(filename, poses, timestamps)

    def _write_result_poses(self):
        self._save_poses(
            filename=f"{self.results_dir}/{self.dataset_sequence}_poses",
            poses=self._calibrate_poses(self.poses),
            timestamps=self._get_frames_timestamps(),
        )

    def _write_gt_poses(self):
        if not self.has_gt:
            return
        self._save_poses(
            filename=f"{self.results_dir}/{self.dataset_sequence}_gt",
            poses=self._calibrate_poses(self.gt_poses),
            timestamps=self._get_frames_timestamps(),
        )

    def _get_fps(self):
        times_nozero = self.times[self.times != 0]
        total_time_s = np.sum(times_nozero) * 1e-9
        return float(times_nozero.shape[0] / total_time_s) if total_time_s > 0 else 0

    def _run_evaluation(self):
        if self.has_gt:
            avg_tra, avg_rot = sequence_error(self.gt_poses, self.poses)
            ate_rot, ate_trans = absolute_trajectory_error(self.gt_poses, self.poses)
            self.results.append(desc="Average Translation Error", units="%", value=avg_tra)
            self.results.append(desc="Average Rotational Error", units="deg/m", value=avg_rot)
            self.results.append(desc="Absolute Trajectory Error (ATE)", units="m", value=ate_trans)
            self.results.append(desc="Absolute Rotational Error (ARE)", units="rad", value=ate_rot)

        fps = self._get_fps()
        avg_fps = int(np.floor(fps))
        avg_ms = int(np.ceil(1e3 / fps)) if fps > 0 else 0
        if avg_fps > 0:
            self.results.append(desc="Average Frequency", units="Hz", value=avg_fps, trunc=True)
            self.results.append(desc="Average Runtime", units="ms", value=avg_ms, trunc=True)
        self.results.append(desc="Number of closures found", units="closures", value=0, trunc=True)

    def _write_log(self):
        if not self.results.empty():
            self.results.log_to_file(
                f"{self.results_dir}/result_metrics.log",
                f"Results for {self.dataset_name} Sequence {self.dataset_sequence}",
            )

    def _write_cfg(self):
        write_config(self.config, os.path.join(self.results_dir, "config.yml"))

    @staticmethod
    def _get_results_dir(out_dir: str):
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_dir = os.path.join(os.path.realpath(out_dir), now)
        latest_dir = os.path.join(os.path.realpath(out_dir), "latest")
        os.makedirs(results_dir, exist_ok=True)
        if os.path.exists(latest_dir) or os.path.islink(latest_dir):
            os.unlink(latest_dir)
        os.symlink(results_dir, latest_dir)
        return results_dir

    def _create_output_dir(self):
        self.results_dir = self._get_results_dir(self.config.out_dir)

    # --- CÁC HÀM EXPORT GIỐNG GENZ ---
    def _write_graph(self):
        graph_file = os.path.join(self.results_dir, "trajectory.g2o")
        poses = self._calibrate_poses(self.poses)
        with open(graph_file, "w") as f:
            for idx, pose in enumerate(poses):
                tx, ty, tz = pose[:3, 3]
                qw, qx, qy, qz = Quaternion(matrix=pose, atol=0.01).elements
                f.write(f"VERTEX_SE3:QUAT {idx} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")
            for idx in range(len(poses) - 1):
                rel = np.linalg.inv(poses[idx]) @ poses[idx + 1]
                tx, ty, tz = rel[:3, 3]
                qw, qx, qy, qz = Quaternion(matrix=rel, atol=0.01).elements
                info = "1 0 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 1 0 1"
                f.write(f"EDGE_SE3:QUAT {idx} {idx+1} {tx} {ty} {tz} {qx} {qy} {qz} {qw} {info}\n")

    def _write_trajectory_plot(self):
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return
        poses = self._calibrate_poses(self.poses)
        plt.figure(figsize=(8, 8))
        plt.plot(poses[:, 0, 3], poses[:, 1, 3], "b-", linewidth=1.5, label="estimated")
        if self.has_gt:
            gt = self._calibrate_poses(self.gt_poses)
            plt.plot(gt[:, 0, 3], gt[:, 1, 3], "g--", linewidth=1.0, label="gt")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.axis("equal")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "trajectory.png"), dpi=300)
        plt.close()

    def _write_local_maps(self):
        local_maps_dir = os.path.join(self.results_dir, "local_maps")
        os.makedirs(local_maps_dir, exist_ok=True)
        np.save(os.path.join(local_maps_dir, "local_map_final.npy"), self.odometry.local_map)
