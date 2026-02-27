// MIT License
//
// Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
// Stachniss.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <vector>
#include <tuple> // Thêm tuple

#include "VoxelHashMap.hpp"

namespace kiss_icp {

struct Registration {
    // [SỬA LẠI] Thêm reg_mode vào cuối hàm khởi tạo
    explicit Registration(int max_num_iteration, double convergence_criterion, int max_num_threads,
                          double adaptive_base, double min_planarity_thr, double max_planarity_thr,
                          int reg_mode, int min_points_pca);

    // [SỬA LẠI] Trả về Tuple (Pose, Planar Pts, Non-Planar Pts)
    std::tuple<Sophus::SE3d, std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>>
    AlignPointsToMap(const std::vector<Eigen::Vector3d> &frame,
                     const VoxelHashMap &voxel_map,
                     const Sophus::SE3d &initial_guess,
                     const double max_correspondence_distance,
                     const double kernel_scale);

    int max_num_iterations_;
    double convergence_criterion_;
    int max_num_threads_;
    
    double adaptive_base_;
    double min_planarity_thr_;
    double max_planarity_thr_;
    
    // [THÊM MỚI] Biến lưu chế độ: 0=Hybrid, 1=Point2Point, 2=Point2Plane
    int reg_mode_; 
    int min_points_pca_;
};
}  // namespace kiss_icp
