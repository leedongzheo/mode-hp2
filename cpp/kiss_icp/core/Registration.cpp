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

inline double square(double x) { return x * x; }

// -------------------- Hybrid correspondences structure --------------------
struct HybridCorrespondence {
    std::vector<Eigen::Vector3d> src_planar, tgt_planar, normals;
    std::vector<Eigen::Vector3d> src_non_planar, tgt_non_planar;
    size_t planar_count = 0, non_planar_count = 0;
};

double ComputeAdaptivePlaharityThreshold(const std::vector<Eigen::Vector3d>& neighbors, 
                                         double base, double min_thr, double max_thr){
    double thr = base * 20.0 / std::max(20.0, static_cast<double>(neighbors.size()));
    return std::clamp(thr, min_thr, max_thr);
}

std::tuple<bool, Eigen::Vector3d> EstimateNormalAndPlanarityC1(
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
    const double planarity = lambda0 / sumlam;

    double adaptive_thr = ComputeAdaptivePlaharityThreshold(neighbors, base, min_thr, max_thr);
    const bool is_planar = planarity < adaptive_thr;
    Eigen::Vector3d normal = evecs.col(0);
    return {is_planar, normal};
}

// -------------------- Parallel TransformPoints --------------------
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

// -------------------- Parallel Hybrid Data Association --------------------
HybridCorrespondence ComputeHybridCorrespondencesParallel(
    const std::vector<Eigen::Vector3d>& source_points,
    const kiss_icp::VoxelHashMap& voxel_map,
    double max_correspondence_distance,
    double base, double min_thr, double max_thr, int reg_mode)
{
    struct LocalBuf {
        std::vector<Eigen::Vector3d> src_planar, tgt_planar, normals;
        std::vector<Eigen::Vector3d> src_non_planar, tgt_non_planar;
        size_t planar_count = 0, non_planar_count = 0;

        void reserve_hint(size_t n) {
            const size_t hint = std::max<size_t>(32, n / 2);
            src_planar.reserve(hint);  tgt_planar.reserve(hint); normals.reserve(hint);
            src_non_planar.reserve(hint); tgt_non_planar.reserve(hint);
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
                const auto& pt = source_points[i];
                auto [closest, neighbors, dist] = voxel_map.GetClosestNeighborAndNeighbors(pt);
                if (dist > max_correspondence_distance) continue;

                if (reg_mode == 1) {
                    buf.src_non_planar.push_back(pt);
                    buf.tgt_non_planar.push_back(closest);
                    buf.non_planar_count++;
                    continue; 
                }

                if (neighbors.size() >= 20) {
                    auto [is_planar, normal] = EstimateNormalAndPlanarityC1(neighbors, base, min_thr, max_thr);
                    
                    if (reg_mode == 2) {
                        buf.src_planar.push_back(pt);
                        buf.tgt_planar.push_back(closest);
                        buf.normals.push_back(normal);
                        buf.planar_count++;
                    } 
                    else {
                        if (is_planar) {
                            buf.src_planar.push_back(pt);
                            buf.tgt_planar.push_back(closest);
                            buf.normals.push_back(normal);
                            buf.planar_count++;
                        } else {
                            buf.src_non_planar.push_back(pt);
                            buf.tgt_non_planar.push_back(closest);
                            buf.non_planar_count++;
                        }
                    }
                } else {
                    if (reg_mode != 2) {
                        buf.src_non_planar.push_back(pt);
                        buf.tgt_non_planar.push_back(closest);
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
    out.src_planar.reserve(total_planar);
    out.tgt_planar.reserve(total_planar);
    out.normals.reserve(total_planar);
    out.src_non_planar.reserve(total_nonplanar);
    out.tgt_non_planar.reserve(total_nonplanar);

    for (auto& buf : tls) {
        out.planar_count     += buf.planar_count;
        out.non_planar_count += buf.non_planar_count;

        out.src_planar.insert(out.src_planar.end(), buf.src_planar.begin(), buf.src_planar.end());
        out.tgt_planar.insert(out.tgt_planar.end(), buf.tgt_planar.begin(), buf.tgt_planar.end());
        out.normals.insert(out.normals.end(), buf.normals.begin(), buf.normals.end());

        out.src_non_planar.insert(out.src_non_planar.end(), buf.src_non_planar.begin(), buf.src_non_planar.end());
        out.tgt_non_planar.insert(out.tgt_non_planar.end(), buf.tgt_non_planar.begin(), buf.tgt_non_planar.end());
    }
    return out;
}

// -------------------- Parallel BuildLinearSystem --------------------
std::tuple<Eigen::Matrix6d, Eigen::Vector6d> BuildHybridLinearSystemParallel(
    const HybridCorrespondence& corr, double kernel, double alpha)
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
                a.JTJ.noalias() += alpha * (J.transpose() * (w * J));
                a.JTr.noalias() += alpha * (J.transpose() * (w * residual));
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
                a.JTJ.noalias() += (1.0 - alpha) * (J.transpose() * (w * J));
                a.JTr.noalias() += (1.0 - alpha) * (J.transpose() * (w * rvec));
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
        
        auto corr = ComputeHybridCorrespondencesParallel(
            source, voxel_map, max_distance, 
            adaptive_base_, min_planarity_thr_, max_planarity_thr_, reg_mode_
        );

        double alpha = 0.5;
        if (reg_mode_ == 1) {
            alpha = 0.0; 
        } else if (reg_mode_ == 2) {
            alpha = 1.0; 
        } else {
            const double denom = static_cast<double>(corr.planar_count + corr.non_planar_count);
            alpha = (denom > 0.0) ? static_cast<double>(corr.planar_count) / denom : 0.5;
        }

        // [THÊM MỚI] In log ra màn hình
        std::cout << "[DEBUG ICP] Mode: " << reg_mode_ 
                  << " | Iter: " << j 
                  << " | Alpha: " << alpha 
                  << " | Planar pts: " << corr.planar_count 
                  << " | Non-Planar pts: " << corr.non_planar_count << std::endl;

        auto [JTJ, JTr] = BuildHybridLinearSystemParallel(corr, kernel, alpha);

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
