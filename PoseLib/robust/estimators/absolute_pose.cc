// Copyright (c) 2021, Viktor Larsson
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "absolute_pose.h"

#include "PoseLib/robust/bundle.h"
#include "PoseLib/solvers/gp3p.h"
#include "PoseLib/solvers/p1p2ll.h"
#include "PoseLib/solvers/p2p1ll.h"
#include "PoseLib/solvers/p3ll.h"
#include "PoseLib/solvers/p3p.h"
#include "PoseLib/solvers/p5lp_radial.h"
#include "PoseLib/solvers/up2p.h"

namespace poselib {

void AbsolutePoseEstimator::generate_models(std::vector<CameraPose> *models) {
    sampler.generate_sample(&sample);
    for (size_t k = 0; k < sample_sz; ++k) {
        xs[k] = x[sample[k]].homogeneous().normalized();
        Xs[k] = X[sample[k]];
    }
    p3p(xs, Xs, models);
}

double AbsolutePoseEstimator::score_model(const CameraPose &pose, size_t *inlier_count) const {
    return compute_msac_score(pose, x, X, opt.max_reproj_error * opt.max_reproj_error, inlier_count);
}

void AbsolutePoseEstimator::refine_model(CameraPose *pose) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_reproj_error;
    bundle_opt.max_iterations = 25;

    // TODO: for high outlier scenarios, make a copy of (x,X) and find points close to inlier threshold
    // TODO: experiment with good thresholds for copy vs iterating full point set
    bundle_adjust(x, X, pose, bundle_opt);
}

void AbsolutePoseUprightEstimator::generate_models(std::vector<CameraPose> *models) {
    up2p_sampler.generate_sample(&sample);
    for (size_t k = 0; k < sample_sz; ++k) {
        xs[k] = x[sample[k]].homogeneous().normalized();
        Xs[k] = X[sample[k]];
    }
    up2p(xs, Xs, models);
}

double AbsolutePoseUprightEstimator::score_model(const CameraPose &pose, size_t *inlier_count) {
    std::vector<char> inliers_mask;
    get_inliers(pose, x, X, opt.max_reproj_error * opt.max_reproj_error, &inliers_mask);

    // TODO: rewrite using transform or ranges
    x_inliers.clear();
    X_inliers.clear();
    for (size_t i = 0; i < inliers_mask.size(); ++i) {
        if (inliers_mask[i]) {
            x_inliers.push_back(x[i]);
            X_inliers.push_back(X[i]);
        }
    }
    *inlier_count = x_inliers.size();

    size_t tmp;
    return compute_msac_score(pose, x, X, opt.max_reproj_error * opt.max_reproj_error, &tmp);
}

void AbsolutePoseUprightEstimator::refine_model(CameraPose *pose) {
    RandomSampler p3p_sampler(x_inliers.size(), p3p_sample_sz, opt.seed, opt.progressive_sampling, opt.max_prosac_iterations);
    std::vector<CameraPose> models;
    CameraPose best_model;
    size_t stats_num_inliers = 0;
    double stats_model_score = std::numeric_limits<double>::max();

    for (auto n = 0; n < 10; ++n) {
        models.clear();

        p3p_sampler.generate_sample(&sample);
        for (size_t k = 0; k < sample_sz; ++k) {
            xs[k] = x_inliers[sample[k]].homogeneous().normalized();
            Xs[k] = X_inliers[sample[k]];
        }
        models.clear();
        p3p(xs, Xs, &models);

        // Find best model among candidates
        size_t inlier_count = 0;
        size_t best_minimal_inlier_count = 0;
        double best_minimal_msac_score = std::numeric_limits<double>::max();
        for (size_t i = 0; i < models.size(); ++i) {
            double score_msac = score_model(models[i], &inlier_count);
            bool more_inliers = inlier_count > best_minimal_inlier_count;
            bool better_score = score_msac < best_minimal_msac_score;

            if (more_inliers || better_score) {
                if (more_inliers) {
                    best_minimal_inlier_count = inlier_count;
                }
                if (better_score) {
                    best_minimal_msac_score = score_msac;
                }

                // check if we should update best model already
                if (score_msac < stats_model_score) {
                    stats_model_score = score_msac;
                    best_model = models[i];
                    stats_num_inliers = inlier_count;
                }
            }
        }
    }

    // Outlier sample or wrong gravity prior
    if (stats_num_inliers < 1.2 * x_inliers.size()) {
        return;
    }

    *pose = best_model;

    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_reproj_error;
    bundle_opt.max_iterations = 25;

    // TODO: for high outlier scenarios, make a copy of (x,X) and find points close to inlier threshold
    // TODO: experiment with good thresholds for copy vs iterating full point set
    bundle_adjust(x, X, pose, bundle_opt);
}

void GeneralizedAbsolutePoseEstimator::generate_models(std::vector<CameraPose> *models) {
    draw_sample(sample_sz, num_pts_camera, &sample, rng);

    for (size_t k = 0; k < sample_sz; ++k) {
        const size_t cam_k = sample[k].first;
        const size_t pt_k = sample[k].second;
        ps[k] = camera_centers[cam_k];
        xs[k] = rig_poses[cam_k].derotate(x[cam_k][pt_k].homogeneous().normalized());
        Xs[k] = X[cam_k][pt_k];
    }
    gp3p(ps, xs, Xs, models);
}

double GeneralizedAbsolutePoseEstimator::score_model(const CameraPose &pose, size_t *inlier_count) const {
    const double sq_threshold = opt.max_reproj_error * opt.max_reproj_error;
    double score = 0;
    *inlier_count = 0;
    size_t cam_inlier_count;
    for (size_t k = 0; k < num_cams; ++k) {
        CameraPose full_pose;
        full_pose.q = quat_multiply(rig_poses[k].q, pose.q);
        full_pose.t = rig_poses[k].rotate(pose.t) + rig_poses[k].t;

        score += compute_msac_score(full_pose, x[k], X[k], sq_threshold, &cam_inlier_count);
        *inlier_count += cam_inlier_count;
    }
    return score;
}

void GeneralizedAbsolutePoseEstimator::refine_model(CameraPose *pose) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_reproj_error;
    bundle_opt.max_iterations = 25;
    generalized_bundle_adjust(x, X, rig_poses, pose, bundle_opt);
}

void AbsolutePosePointLineEstimator::generate_models(std::vector<CameraPose> *models) {
    draw_sample(sample_sz, num_data, &sample, rng);

    size_t pt_idx = 0;
    size_t line_idx = 0;
    for (size_t k = 0; k < sample_sz; ++k) {
        size_t idx = sample[k];
        if (idx < points2D.size()) {
            // we sampled a point correspondence
            xs[pt_idx] = points2D[idx].homogeneous();
            xs[pt_idx].normalize();
            Xs[pt_idx] = points3D[idx];
            pt_idx++;
        } else {
            // we sampled a line correspondence
            idx -= points2D.size();
            ls[line_idx] = lines2D[idx].x1.homogeneous().cross(lines2D[idx].x2.homogeneous());
            ls[line_idx].normalize();
            Cs[line_idx] = lines3D[idx].X1;
            Vs[line_idx] = lines3D[idx].X2 - lines3D[idx].X1;
            Vs[line_idx].normalize();
            line_idx++;
        }
    }

    if (pt_idx == 3 && line_idx == 0) {
        p3p(xs, Xs, models);
    } else if (pt_idx == 2 && line_idx == 1) {
        p2p1ll(xs, Xs, ls, Cs, Vs, models);
    } else if (pt_idx == 1 && line_idx == 2) {
        p1p2ll(xs, Xs, ls, Cs, Vs, models);
    } else if (pt_idx == 0 && line_idx == 3) {
        p3ll(ls, Cs, Vs, models);
    }
}

double AbsolutePosePointLineEstimator::score_model(const CameraPose &pose, size_t *inlier_count) const {
    size_t point_inliers, line_inliers;
    double score_pt =
        compute_msac_score(pose, points2D, points3D, opt.max_reproj_error * opt.max_reproj_error, &point_inliers);
    double score_l =
        compute_msac_score(pose, lines2D, lines3D, opt.max_epipolar_error * opt.max_epipolar_error, &line_inliers);
    *inlier_count = point_inliers + line_inliers;
    return score_pt + score_l;
}

void AbsolutePosePointLineEstimator::refine_model(CameraPose *pose) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_reproj_error;
    bundle_opt.max_iterations = 25;

    BundleOptions line_bundle_opt;
    line_bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    line_bundle_opt.loss_scale = opt.max_epipolar_error;

    bundle_adjust(points2D, points3D, lines2D, lines3D, pose, bundle_opt, line_bundle_opt);
}

void Radial1DAbsolutePoseEstimator::generate_models(std::vector<CameraPose> *models) {
    sampler.generate_sample(&sample);
    for (size_t k = 0; k < sample_sz; ++k) {
        xs[k] = x[sample[k]].normalized();
        Xs[k] = X[sample[k]];
    }
    p5lp_radial(xs, Xs, models);
}

double Radial1DAbsolutePoseEstimator::score_model(const CameraPose &pose, size_t *inlier_count) const {
    return compute_msac_score_1D_radial(pose, x, X, opt.max_reproj_error * opt.max_reproj_error, inlier_count);
}

void Radial1DAbsolutePoseEstimator::refine_model(CameraPose *pose) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_reproj_error;
    bundle_opt.max_iterations = 25;

    // TODO: for high outlier scenarios, make a copy of (x,X) and find points close to inlier threshold
    // TODO: experiment with good thresholds for copy vs iterating full point set

    bundle_adjust_1D_radial(x, X, pose, bundle_opt);
}

} // namespace poselib