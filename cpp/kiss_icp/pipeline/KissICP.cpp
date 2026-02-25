#include "KissICP.hpp"

#include <Eigen/Core>
#include <vector>

#include "kiss_icp/core/Preprocessing.hpp"
#include "kiss_icp/core/Registration.hpp"
#include "kiss_icp/core/VoxelHashMap.hpp"

namespace kiss_icp::pipeline {

KissICP::Vector3dVectorTuple KissICP::RegisterFrame(const std::vector<Eigen::Vector3d> &frame,
                                                    const std::vector<double> &timestamps) {
    // Preprocess the input cloud
    const auto &preprocessed_frame = preprocessor_.Preprocess(frame, timestamps, last_delta_);

    // Voxelize
    const auto &[source, frame_downsample] = Voxelize(preprocessed_frame);

    // Get adaptive_threshold
    const double sigma = adaptive_threshold_.ComputeThreshold();

    // Compute initial_guess for ICP
    const auto initial_guess = last_pose_ * last_delta_;

    // --- [SỬA QUAN TRỌNG 1] ---
    // Run ICP: Hứng lấy tuple 3 phần tử trả về từ AlignPointsToMap
    const auto &[new_pose, planar_points, non_planar_points] = registration_.AlignPointsToMap(
                                                         source,         // frame
                                                         local_map_,     // voxel_map
                                                         initial_guess,  // initial_guess
                                                         3.0 * sigma,    // max_correspondence_dist
                                                         sigma);         // kernel
    // --------------------------

    // Compute the difference between the prediction and the actual estimate
    const auto model_deviation = initial_guess.inverse() * new_pose;

    // Update step: threshold, local map, delta, and the last pose
    adaptive_threshold_.UpdateModelDeviation(model_deviation);
    local_map_.Update(frame_downsample, new_pose);
    last_delta_ = last_pose_.inverse() * new_pose;
    last_pose_ = new_pose;

    // --- [SỬA QUAN TRỌNG 2] ---
    // Trả về 2 mảng điểm Planar và Non-Planar để truyền qua Python lên Visualizer
    return {planar_points, non_planar_points};
    // --------------------------
}

KissICP::Vector3dVectorTuple KissICP::Voxelize(const std::vector<Eigen::Vector3d> &frame) const {
    const auto voxel_size = config_.voxel_size;
    const auto frame_downsample = kiss_icp::VoxelDownsample(frame, voxel_size * 0.5);
    const auto source = kiss_icp::VoxelDownsample(frame_downsample, voxel_size * 1.5);
    return {source, frame_downsample};
}

void KissICP::Reset() {
    last_pose_ = Sophus::SE3d();
    last_delta_ = Sophus::SE3d();

    // Clear the local map
    local_map_.Clear();

    // Reset adaptive threshold (it will start fresh)
    adaptive_threshold_ =
        AdaptiveThreshold(config_.initial_threshold, config_.min_motion_th, config_.max_range);
}

}  // namespace kiss_icp::pipeline