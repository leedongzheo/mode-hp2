#pragma once

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <tuple>
#include <vector>

#include "kiss_icp/core/Preprocessing.hpp"
#include "kiss_icp/core/Registration.hpp"
#include "kiss_icp/core/Threshold.hpp"
#include "kiss_icp/core/VoxelHashMap.hpp"

namespace kiss_icp::pipeline {

struct KISSConfig {
    // map params
    double voxel_size = 1.0;
    double max_range = 100.0;
    double min_range = 0.0;
    int max_points_per_voxel = 20;

    // th parms
    double min_motion_th = 0.1;
    double initial_threshold = 2.0;

    // registration params
    int max_num_iterations = 500;
    double convergence_criterion = 0.0001;
    int max_num_threads = 0;

    // Motion compensation
    bool deskew = true;

    // Adaptive Planarity Threshold Params
    double adaptive_base = 0.06;
    double min_planarity_thr = 0.001;
    double max_planarity_thr = 0.2;
    
    // Biến chọn chế độ: 0=Hybrid, 1=Point2Point, 2=Point2Plane
    int reg_mode = 0; 

    // [THÊM MỚI] Khai báo min_points_pca ở đây, giá trị mặc định là 5
    int min_points_pca = 5; 
};

class KissICP {
public:
    using Vector3dVector = std::vector<Eigen::Vector3d>;
    using Vector3dVectorTuple = std::tuple<Vector3dVector, Vector3dVector>;

public:
    explicit KissICP(const KISSConfig &config)
        : config_(config),
          preprocessor_(config.max_range, config.min_range, config.deskew, config.max_num_threads),
          registration_(
              config.max_num_iterations, 
              config.convergence_criterion, 
              config.max_num_threads,
              config.adaptive_base,       
              config.min_planarity_thr,   
              config.max_planarity_thr,
              config.reg_mode,
              config.min_points_pca), // [THÊM MỚI] Truyền min_points_pca vào Registration
          local_map_(config.voxel_size, config.max_range, config.max_points_per_voxel),
          adaptive_threshold_(config.initial_threshold, config.min_motion_th, config.max_range) {}

public:
    Vector3dVectorTuple RegisterFrame(const std::vector<Eigen::Vector3d> &frame,
                                      const std::vector<double> &timestamps);
    Vector3dVectorTuple Voxelize(const std::vector<Eigen::Vector3d> &frame) const;

    std::vector<Eigen::Vector3d> LocalMap() const { return local_map_.Pointcloud(); };

    const VoxelHashMap &VoxelMap() const { return local_map_; };
    VoxelHashMap &VoxelMap() { return local_map_; };

    const Sophus::SE3d &pose() const { return last_pose_; }
    Sophus::SE3d &pose() { return last_pose_; }

    const Sophus::SE3d &delta() const { return last_delta_; }
    Sophus::SE3d &delta() { return last_delta_; }
    void Reset();

private:
    Sophus::SE3d last_pose_;
    Sophus::SE3d last_delta_;

    // KISS-ICP pipeline modules
    KISSConfig config_;
    Preprocessor preprocessor_;
    Registration registration_;
    VoxelHashMap local_map_;
    AdaptiveThreshold adaptive_threshold_;
};

}  // namespace kiss_icp::pipeline
