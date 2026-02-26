#include "Registration.hpp"

#include <Eigen/Eigenvalues>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/global_control.h>

#include <algorithm>
#include <cmath>
#include <tuple>
#include <iostream>

#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

#include "VoxelHashMap.hpp"
#include "VoxelUtils.hpp"

namespace Eigen {
using Matrix6d   = Eigen::Matrix<double, 6, 6>;
using Matrix3_6d = Eigen::Matrix<double, 3, 6>;
using Vector6d   = Eigen::Matrix<double, 6, 1>;
}  // namespace Eigen

namespace {

// Khai báo hằng số
constexpr size_t MIN_POINTS_PCA = 20;
constexpr double MIN_POINTS_PCA_D = static_cast<double>(MIN_POINTS_PCA);

// [ĐỘT PHÁ] Ngưỡng Z cục bộ (Local Z) để tách mặt đất.
// Xe KITTI có LiDAR cao 1.73m. Ta lấy -1.6m để bù trừ nhún nhảy.
constexpr double GROUND_Z_THRESHOLD = -1.6;

inline double square(double x) { return x * x; }

// -------------------- Hybrid correspondences structure --------------------
struct HybridCorrespondence {
    std::vector<Eigen::Vector3d> src_planar, tgt_planar, normals;
    std::vector<Eigen::Vector3d> src_non_planar, tgt_non_planar;
    
    // Lưu trữ trọng số (alpha) RIÊNG BIỆT cho từng điểm
    std::vector<double> weights_planar;
    std::vector<double> weights_non_planar;
    
    size_t planar_count = 0, non_planar_count = 0;
};

double ComputeAdaptivePlaharityThreshold(const std::vector<Eigen::Vector3d>& neighbors, 
                                         double base, double min_thr, double max_thr){
    double thr = base * MIN_POINTS_PCA_D / std::max(MIN_POINTS_PCA_D, static_cast<double>(neighbors.size()));
    return std::clamp(thr, min_thr, max_thr);
}

std::tuple<bool, Eigen::Vector3d, double> EstimateNormalAndPlanarityC1(
    const std::vector<Eigen::Vector3d>& neighbors,
    double base, double min_thr, double max_thr)
{
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const auto& pt : neighbors) mean += pt;
    mean /= static_cast<double>(neighbors.size());

    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (const auto& pt : neighbors) {
        Eigen::Vector3d d = pt - mean;
        cov.noalias() += d * d.transpose();
    }
    cov /= static_cast<double>(neighbors.size());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov);
    const auto& evals = eig.eigenvalues();
    const auto& evecs = eig.eigenvectors();

    const double lambda0 = evals(0);
    const double sumlam  = evals(0) + evals(1) + evals(2) + 1e-12;
    const double planarity = lambda0 / sumlam; // Nằm trong khoảng [0, 1/3]

    double adaptive_thr = ComputeAdaptivePlaharityThreshold(neighbors, base, min_thr, max_thr);
    const bool is_planar = planarity < adaptive_thr;
    Eigen::Vector3d normal = evecs.col(0);
    
    // Tính trọng số Point-wise. 
    double w_plane = std::clamp(1.0 - 3.0 * planarity, 0.0, 1.0);
    
    return {is_planar, normal, w_plane};
}

void TransformPoints(const Sophus::SE3d &T, std::vector<Eigen::Vector3d> &points) {
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, points.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                points[i] = T * points[i];
            }
        }
    );
}

// [SỬA LỚN] Thêm tham số `local_frame` chứa tọa độ nguyên bản từ xe
HybridCorrespondence ComputeHybridCorrespondencesParallel(
    const std::vector<Eigen::Vector3d>& source_points,
    const std::vector<Eigen::Vector3d>& local_frame, 
    const kiss_icp::VoxelHashMap& voxel_map,
    double max_correspondence_distance,
    double base, double min_thr, double max_thr, int reg_mode)
{
    struct LocalBuf {
        std::vector<Eigen::Vector3d> src_planar, tgt_planar, normals;
        std::vector<Eigen::Vector3d> src_non_planar, tgt_non_planar;
        std::vector<double> weights_planar, weights_non_planar;
        size_t planar_count = 0, non_planar_count = 0;

        void reserve_hint(size_t n) {
            const size_t hint = std::max<size_t>(32, n / 2);
            src_planar.reserve(hint);  tgt_planar.reserve(hint); normals.reserve(hint);
            src_non_planar.reserve(hint); tgt_non_planar.reserve(hint);
            weights_planar.reserve(hint); weights_non_planar.reserve(hint);
        }
    };

    tbb::enumerable_thread_specific<LocalBuf> tls;
    for (auto it = tls.begin(); it != tls.end(); ++it) {
        it->reserve_hint(source_points.size());
    }

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, source_points.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            auto& buf = tls.local();
            for (size_t i = r.begin(); i != r.end(); ++i) {
                const auto& pt = source_points[i]; // Tọa độ Map (để tìm kiếm)
                const double local_z = local_frame[i].z(); // Tọa độ Gầm xe (để cắt mặt đất)

                auto [closest, neighbors, dist] = voxel_map.GetClosestNeighborAndNeighbors(pt);
                if (dist > max_correspondence_distance) continue;

                if (reg_mode == 1) { // Ép Point-to-Point
                    buf.src_non_planar.push_back(pt);
                    buf.tgt_non_planar.push_back(closest);
                    buf.weights_non_planar.push_back(1.0);
                    buf.non_planar_count++;
                    continue; 
                }

                // ==========================================
                // 1. CHẶT CỤT TRỤC Z CỤC BỘ (GROUND Z-CUT)
                // ==========================================
                if (local_z < GROUND_Z_THRESHOLD) {
                    // Ép làm mặt đất 100%, bỏ qua PCA cho nhẹ máy
                    buf.src_planar.push_back(pt);
                    buf.tgt_planar.push_back(closest);
                    // Giả định mặt đất luôn hướng thẳng lên trời (trục Z)
                    buf.normals.push_back(Eigen::Vector3d(0.0, 0.0, 1.0)); 
                    buf.weights_planar.push_back(1.0); // Trọng số tối đa
                    buf.planar_count++;
                    continue; // Xong điểm này, chuyển qua điểm tiếp theo
                }

                // ==========================================
                // 2. XỬ LÝ VẬT THỂ & HÀNH LANG (HYBRID POINT-WISE)
                // ==========================================
                if (neighbors.size() >= MIN_POINTS_PCA) {
                    auto [is_planar, normal, w_plane] = EstimateNormalAndPlanarityC1(neighbors, base, min_thr, max_thr);
                    
                    if (reg_mode == 2) { // Ép Point-to-Plane
                        buf.src_planar.push_back(pt);
                        buf.tgt_planar.push_back(closest);
                        buf.normals.push_back(normal);
                        buf.weights_planar.push_back(1.0);
                        buf.planar_count++;
                    } 
                    else { // Hybrid Mode
                        if (is_planar) {
                            // Tường, Thành cầu, Biển báo giao thông
                            buf.src_planar.push_back(pt);
                            buf.tgt_planar.push_back(closest);
                            buf.normals.push_back(normal); // Lấy pháp tuyến ngang đâm ra từ tường
                            buf.weights_planar.push_back(w_plane); 
                            buf.planar_count++;
                        } else {
                            // Cây cối, Người đi bộ, Cột điện tròn
                            buf.src_non_planar.push_back(pt);
                            buf.tgt_non_planar.push_back(closest);
                            buf.weights_non_planar.push_back(1.0 - w_plane); 
                            buf.non_planar_count++;
                        }
                    }
                } else {
                    if (reg_mode != 2) {
                        buf.src_non_planar.push_back(pt);
                        buf.tgt_non_planar.push_back(closest);
                        buf.weights_non_planar.push_back(1.0);
                        buf.non_planar_count++;
                    }
                }
            }
        }
    );

    HybridCorrespondence out;
    size_t total_planar = 0, total_nonplanar = 0;
    for (auto& buf : tls) {
        total_planar    += buf.planar_count;
        total_nonplanar += buf.non_planar_count;
    }
    out.src_planar.reserve(total_planar); out.tgt_planar.reserve(total_planar); out.normals.reserve(total_planar); out.weights_planar.reserve(total_planar);
    out.src_non_planar.reserve(total_nonplanar); out.tgt_non_planar.reserve(total_nonplanar); out.weights_non_planar.reserve(total_nonplanar);

    for (auto& buf : tls) {
        out.planar_count     += buf.planar_count;
        out.non_planar_count += buf.non_planar_count;

        out.src_planar.insert(out.src_planar.end(), buf.src_planar.begin(), buf.src_planar.end());
        out.tgt_planar.insert(out.tgt_planar.end(), buf.tgt_planar.begin(), buf.tgt_planar.end());
        out.normals.insert(out.normals.end(), buf.normals.begin(), buf.normals.end());
        out.weights_planar.insert(out.weights_planar.end(), buf.weights_planar.begin(), buf.weights_planar.end());

        out.src_non_planar.insert(out.src_non_planar.end(), buf.src_non_planar.begin(), buf.src_non_planar.end());
        out.tgt_non_planar.insert(out.tgt_non_planar.end(), buf.tgt_non_planar.begin(), buf.tgt_non_planar.end());
        out.weights_non_planar.insert(out.weights_non_planar.end(), buf.weights_non_planar.begin(), buf.weights_non_planar.end());
    }
    return out;
}

std::tuple<Eigen::Matrix6d, Eigen::Vector6d> BuildHybridLinearSystemParallel(
    const HybridCorrespondence& corr, double kernel)
{
    struct Accum {
        Eigen::Matrix<double,6,6> JTJ;
        Eigen::Matrix<double,6,1> JTr;
        Accum() { JTJ.setZero(); JTr.setZero(); }
        Accum(Accum&, tbb::split) { JTJ.setZero(); JTr.setZero(); }
        void join(const Accum& other) {
            JTJ.noalias() += other.JTJ;
            JTr.noalias() += other.JTr;
        }
    };

    const double kernel2 = kernel * kernel;

    Accum acc_plane = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, corr.src_planar.size()),
        Accum{},
        [&](const tbb::blocked_range<size_t>& r, Accum a)->Accum {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                const Eigen::Vector3d& n = corr.normals[i];
                const double residual = (corr.src_planar[i] - corr.tgt_planar[i]).dot(n);

                Eigen::Matrix<double,1,6> J;
                J.block<1,3>(0,0) = n.transpose();
                J.block<1,3>(0,3) = (corr.src_planar[i].cross(n)).transpose();

                const double w = kernel2 / square(kernel + residual * residual);
                const double point_alpha = corr.weights_planar[i]; // Lấy trọng số cá nhân
                
                a.JTJ.noalias() += point_alpha * (J.transpose() * (w * J));
                a.JTr.noalias() += point_alpha * (J.transpose() * (w * residual));
            }
            return a;
        },
        [](Accum a, const Accum& b)->Accum { a.join(b); return a; }
    );

    Accum acc_point = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, corr.src_non_planar.size()),
        Accum{},
        [&](const tbb::blocked_range<size_t>& r, Accum a)->Accum {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                const Eigen::Vector3d rvec = corr.src_non_planar[i] - corr.tgt_non_planar[i];
                Eigen::Matrix<double,3,6> J;
                J.block<3,3>(0,0).setIdentity();
                J.block<3,3>(0,3) = -Sophus::SO3d::hat(corr.src_non_planar[i]);

                const double w = kernel2 / square(kernel + rvec.squaredNorm());
                const double point_alpha = corr.weights_non_planar[i]; // Lấy trọng số cá nhân
                
                a.JTJ.noalias() += point_alpha * (J.transpose() * (w * J));
                a.JTr.noalias() += point_alpha * (J.transpose() * (w * rvec));
            }
            return a;
        },
        [](Accum a, const Accum& b)->Accum { a.join(b); return a; }
    );

    Eigen::Matrix6d JTJ = acc_plane.JTJ + acc_point.JTJ;
    Eigen::Vector6d JTr = acc_plane.JTr + acc_point.JTr;
    return {JTJ, JTr};
}

}  // namespace (anon)

// ============================ kiss_icp namespace ============================
namespace kiss_icp {

Registration::Registration(int max_num_iteration, double convergence_criterion, int max_num_threads,
                           double adaptive_base, double min_planarity_thr, double max_planarity_thr,
                           int reg_mode)
    : max_num_iterations_(max_num_iteration),
      convergence_criterion_(convergence_criterion),
      adaptive_base_(adaptive_base),
      min_planarity_thr_(min_planarity_thr),
      max_planarity_thr_(max_planarity_thr),
      reg_mode_(reg_mode) {

    if (max_num_threads > 0) {
        static tbb::global_control gc(tbb::global_control::max_allowed_parallelism, max_num_threads);
        (void)gc; 
    }
}

std::tuple<Sophus::SE3d, std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>> 
Registration::AlignPointsToMap(const std::vector<Eigen::Vector3d> &frame,
                               const VoxelHashMap &voxel_map,
                               const Sophus::SE3d &initial_guess,
                               double max_distance,
                               double kernel)
{
    std::vector<Eigen::Vector3d> final_planar;
    std::vector<Eigen::Vector3d> final_non_planar;

    if (voxel_map.Empty()) return {initial_guess, final_planar, final_non_planar};

    std::vector<Eigen::Vector3d> source = frame;
    TransformPoints(initial_guess, source);

    Sophus::SE3d T_icp;
    for (int j = 0; j < max_num_iterations_; ++j) {
        
        // [SỬA ĐỔI] Truyền CẢ `source` (Global) và `frame` (Local) vào hàm
        auto corr = ComputeHybridCorrespondencesParallel(
            source, frame, voxel_map, max_distance, 
            adaptive_base_, min_planarity_thr_, max_planarity_thr_, reg_mode_
        );

        auto [JTJ, JTr] = BuildHybridLinearSystemParallel(corr, kernel);

        Eigen::Vector6d dx = JTJ.ldlt().solve(-JTr);
        Sophus::SE3d delta = Sophus::SE3d::exp(dx);

        TransformPoints(delta, source); 
        T_icp = delta * T_icp;

        if (dx.norm() < convergence_criterion_ || j == max_num_iterations_ - 1) {
            final_planar = corr.src_planar;
            final_non_planar = corr.src_non_planar;
            break;
        }
    }
    return {T_icp * initial_guess, final_planar, final_non_planar};
}

}  // namespace kiss_icp
