#pragma once

#include <tsl/robin_map.h>

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <vector>

#include "VoxelUtils.hpp"

namespace kiss_icp {
struct VoxelHashMap {
    explicit VoxelHashMap(double voxel_size, double max_distance, unsigned int max_points_per_voxel)
        : voxel_size_(voxel_size),
          max_distance_(max_distance),
          max_points_per_voxel_(max_points_per_voxel) {}

    inline void Clear() { map_.clear(); }
    inline bool Empty() const { return map_.empty(); }
    void Update(const std::vector<Eigen::Vector3d> &points, const Eigen::Vector3d &origin);
    void Update(const std::vector<Eigen::Vector3d> &points, const Sophus::SE3d &pose);
    void AddPoints(const std::vector<Eigen::Vector3d> &points);
    void RemovePointsFarFromLocation(const Eigen::Vector3d &origin);
    std::vector<Eigen::Vector3d> Pointcloud() const;
    std::tuple<Eigen::Vector3d, double> GetClosestNeighbor(const Eigen::Vector3d &query) const;

    double voxel_size_;
    double max_distance_;
    unsigned int max_points_per_voxel_;
    tsl::robin_map<Voxel, std::vector<Eigen::Vector3d>> map_;
    std::tuple<Eigen::Vector3d, std::vector<Eigen::Vector3d>, double>GetClosestNeighborAndNeighbors(const Eigen::Vector3d &query) const;
};
}  // namespace kiss_icp