#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Core>
#include <Eigen/Geometry> 
#include <vector>
#include <algorithm>
#include <cmath>

// Quan trọng: Include header chứa class KissICP Pipeline
#include "kiss_icp/pipeline/KissICP.hpp" 
#include "kiss_icp/core/Preprocessing.hpp" // Cho hàm VoxelDownsample global
#include "kiss_icp/metrics/Metrics.hpp"
#include "stl_vector_eigen.h"

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector3d>);

namespace kiss_icp {
PYBIND11_MODULE(kiss_icp_pybind, m) {
    // Helper để chuyển đổi vector Eigen sang Numpy nhanh chóng (như GenZ)
    pybind_eigen_vector_of_vector<Eigen::Vector3d>(
        m, "_Vector3dVector", "std::vector<Eigen::Vector3d>",
        py::py_array_to_vectors_double<Eigen::Vector3d>);

    // 1. Binding Config Struct (Giống _GenZConfig)
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
        // [THÊM MỚI] Expose 3 tham số Adaptive Threshold cho Python
        .def_readwrite("adaptive_base", &pipeline::KISSConfig::adaptive_base)
        .def_readwrite("min_planarity_thr", &pipeline::KISSConfig::min_planarity_thr)
        .def_readwrite("max_planarity_thr", &pipeline::KISSConfig::max_planarity_thr)
        // [THÊM MỚI] Expose biến reg_mode cho Python
        .def_readwrite("reg_mode", &pipeline::KISSConfig::reg_mode);

    // 2. Binding Pipeline Class (Giống _GenZICP)
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
        });

    // 3. Global Utility Functions (Giống GenZ)
    
    // Hàm Voxel Downsample tiện ích
    m.def("_voxel_down_sample", &VoxelDownsample, "frame"_a, "voxel_size"_a);

    // Hàm Correct KITTI Scan (Dùng Lambda)
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
