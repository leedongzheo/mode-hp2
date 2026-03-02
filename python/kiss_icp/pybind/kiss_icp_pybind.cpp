#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Core>
#include <Eigen/Geometry> 
#include <vector>
#include <algorithm>
#include <cmath>
#include <sophus/se3.hpp>

// Quan trọng: Include header chứa class KissICP Pipeline
#include "kiss_icp/pipeline/KissICP.hpp" 
#include "kiss_icp/core/Preprocessing.hpp" // Cho class Preprocessor và hàm VoxelDownsample
#include "kiss_icp/core/VoxelHashMap.hpp"  // Cho class VoxelHashMap
#include "kiss_icp/metrics/Metrics.hpp"
#include "stl_vector_eigen.h"

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector3d>);

namespace kiss_icp {
PYBIND11_MODULE(kiss_icp_pybind, m) {
    // Helper để chuyển đổi vector Eigen sang Numpy nhanh chóng
    pybind_eigen_vector_of_vector<Eigen::Vector3d>(
        m, "_Vector3dVector", "std::vector<Eigen::Vector3d>",
        py::py_array_to_vectors_double<Eigen::Vector3d>);

    // =========================================================================
    // 1. Binding cho VoxelHashMap (Dành cho mapping.py và slam.py)
    // =========================================================================
    py::class_<VoxelHashMap> internal_map(m, "_VoxelHashMap", "VoxelHashMap Binding");
    internal_map
        .def(py::init<double, double, unsigned int>(), 
             "voxel_size"_a, "max_distance"_a, "max_points_per_voxel"_a)
        .def("_clear", &VoxelHashMap::Clear)
        .def("_empty", &VoxelHashMap::Empty)
        .def("_update", 
            [](VoxelHashMap &self, const std::vector<Eigen::Vector3d> &points, const Eigen::Matrix4d &T) {
                Sophus::SE3d pose(T);
                self.Update(points, pose);
            }, 
            "points"_a, "pose"_a)
        .def("_add_points", &VoxelHashMap::AddPoints, "points"_a)
        .def("_remove_far_away_points", &VoxelHashMap::RemovePointsFarFromLocation, "origin"_a)
        .def("_point_cloud", &VoxelHashMap::Pointcloud);

    // =========================================================================
    // [THÊM MỚI] 2. Binding cho Preprocessor (Dành cho preprocess.py và SLAM global mapping)
    // =========================================================================
    py::class_<Preprocessor> internal_preprocessor(m, "_Preprocessor", "Preprocessor Binding");
    internal_preprocessor
        .def(py::init<double, double, bool, int>(), 
             "max_range"_a, "min_range"_a, "deskew"_a, "max_num_threads"_a)
        .def(
            "_preprocess",
            [](const Preprocessor &self, const std::vector<Eigen::Vector3d> &points,
               const std::vector<double> &timestamps, const Eigen::Matrix4d &relative_motion) {
                Sophus::SE3d motion(relative_motion); // Ép kiểu từ Numpy(Matrix4d) sang Sophus::SE3d
                return self.Preprocess(points, timestamps, motion);
            },
            "points"_a, "timestamps"_a, "relative_motion"_a);

    // =========================================================================
    // 3. Binding Config Struct
    // =========================================================================
    py::class_<pipeline::KISSConfig>(m, "_KISSConfig")
        .def(py::init<>())
        .def_readwrite("voxel_size", &pipeline::KISSConfig::voxel_size)
        .def_readwrite("max_range", &pipeline::KISSConfig::max_range)
        .def_readwrite("min_range", &pipeline::KISSConfig::min_range)
        .def_readwrite("max_points_per_voxel", &pipeline::KISSConfig::max_points_per_voxel)
        .def_readwrite("min_motion_th", &pipeline::KISSConfig::min_motion_th)
        .def_readwrite("initial_threshold", &pipeline::KISSConfig::initial_threshold)
        .def_readwrite("max_num_iterations", &pipeline::KISSConfig::max_num_iterations)
        .def_readwrite("convergence_criterion", &pipeline::KISSConfig::convergence_criterion)
        .def_readwrite("max_num_threads", &pipeline::KISSConfig::max_num_threads)
        .def_readwrite("deskew", &pipeline::KISSConfig::deskew)
        .def_readwrite("adaptive_base", &pipeline::KISSConfig::adaptive_base)
        .def_readwrite("min_planarity_thr", &pipeline::KISSConfig::min_planarity_thr)
        .def_readwrite("max_planarity_thr", &pipeline::KISSConfig::max_planarity_thr)
        .def_readwrite("reg_mode", &pipeline::KISSConfig::reg_mode)
        .def_readwrite("min_points_pca", &pipeline::KISSConfig::min_points_pca);

    // =========================================================================
    // 4. Binding Pipeline Class
    // =========================================================================
    py::class_<pipeline::KissICP>(m, "_KissICP")
        .def(py::init<const pipeline::KISSConfig &>(), "config"_a)
        .def("_register_frame", &pipeline::KissICP::RegisterFrame, "frame"_a, "timestamps"_a)
        .def("_local_map", &pipeline::KissICP::LocalMap)
        .def("_voxel_down_sample", &pipeline::KissICP::Voxelize, "frame"_a)
        .def("_reset", &pipeline::KissICP::Reset)
        // Helper: Trả về Pose cuối cùng dưới dạng Matrix4d (Numpy) thay vì Sophus
        .def("_last_pose", [](pipeline::KissICP &self) {
            return self.pose().matrix();
        })
        // Helper: Trả về Delta (Vận tốc)
        .def("_last_delta", [](pipeline::KissICP &self) {
            return self.delta().matrix();
        })
        // CÁC HÀM BỔ SUNG ĐỂ PROXY TƯƠNG TÁC VỚI SLAM
        .def("_clear_local_map", [](pipeline::KissICP &self) {
            self.VoxelMap().Clear(); 
        })
        .def("_add_points_to_map", [](pipeline::KissICP &self, const std::vector<Eigen::Vector3d> &points) {
            self.VoxelMap().AddPoints(points); 
        })
        .def("_set_last_pose", [](pipeline::KissICP &self, const Eigen::Matrix4d &T) {
            self.pose() = Sophus::SE3d(T); 
        });

    // =========================================================================
    // 5. Global Utility Functions
    // =========================================================================
    m.def("_voxel_down_sample", &VoxelDownsample, "frame"_a, "voxel_size"_a);

    m.def(
        "_correct_kitti_scan",
        [](const std::vector<Eigen::Vector3d> &frame) {
            constexpr double VERTICAL_ANGLE_OFFSET = (0.205 * M_PI) / 180.0;
            std::vector<Eigen::Vector3d> frame_ = frame;
            std::transform(frame_.cbegin(), frame_.cend(), frame_.begin(), [&](const auto pt) {
                const Eigen::Vector3d rotationVector = pt.cross(Eigen::Vector3d(0., 0., 1.));
                return Eigen::AngleAxisd(VERTICAL_ANGLE_OFFSET, rotationVector.normalized()) * pt;
            });
            return frame_;
        },
        "frame"_a);

    // Metrics
    m.def("_kitti_seq_error", &metrics::SeqError, "gt_poses"_a, "results_poses"_a);
    m.def("_absolute_trajectory_error", &metrics::AbsoluteTrajectoryError, "gt_poses"_a,
          "results_poses"_a);
}

} // namespace kiss_icp
