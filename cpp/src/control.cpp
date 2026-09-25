// Implementation of the control stack — a faithful C++17 port of
// fsd/control/{pid,lateral,longitudinal,mpc,controller}.py.
#include "fsd/control.hpp"

#include <cmath>
#include <cstddef>
#include <limits>

namespace fsd {
namespace {

/// Extract path coordinates and per-point headings (heading from successive
/// positions — more robust than trusting wp.yaw, which interpolators
/// occasionally leave at 0).
void path_geometry(const Trajectory& traj, std::vector<double>& px,
                   std::vector<double>& py, std::vector<double>& headings) {
    const std::size_t n = traj.points.size();
    px.resize(n);
    py.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        px[i] = traj.points[i].x;
        py[i] = traj.points[i].y;
    }
    headings.resize(n);
    for (std::size_t i = 0; i + 1 < n; ++i)
        headings[i] = std::atan2(py[i + 1] - py[i], px[i + 1] - px[i]);
    headings[n - 1] = n > 1 ? headings[n - 2] : 0.0;
}

/// Index of the nearest path point to (x, y).
std::size_t argmin_d2(const std::vector<double>& px,
                      const std::vector<double>& py, double x, double y) {
    std::size_t best = 0;
    double best_d2 = kInf;
    for (std::size_t i = 0; i < px.size(); ++i) {
        const double dx = px[i] - x, dy = py[i] - y;
        const double d2 = dx * dx + dy * dy;
        if (d2 < best_d2) {
            best_d2 = d2;
            best = i;
        }
    }
    return best;
}

}  // namespace

// ===================================================================== PID

PIDController::PIDController(double kp_, double ki_, double kd_, double i_min_,
                             double i_max_, double out_min_, double out_max_,
                             double deriv_tau_, std::string name_)
    : kp(kp_),
      ki(ki_),
      kd(kd_),
      i_min(i_min_),
      i_max(i_max_),
      out_min(out_min_),
      out_max(out_max_),
      deriv_tau(std::max(deriv_tau_, 1e-3)),
      name(std::move(name_)) {}

void PIDController::reset() {
    integral_ = 0.0;
    prev_error_ = std::nullopt;
    deriv_ = 0.0;
}

double PIDController::step(double error, double dt) {
    const bool dt_valid = dt > 1e-4 && dt < 1.0;

    // --- derivative on error, low-pass filtered -------------------------- //
    if (dt_valid && prev_error_) {
        const double raw_d = (error - *prev_error_) / dt;
        const double alpha = dt / (dt + deriv_tau);  // 1st-order LP coefficient
        deriv_ += alpha * (raw_d - deriv_);
    } else if (!prev_error_) {
        deriv_ = 0.0;  // avoid kick on first call
    }
    prev_error_ = error;

    const double p = kp * error;
    const double d = kd * deriv_;

    // --- tentative output, then conditional integration ------------------ //
    double u_unsat = p + ki * integral_ + d;
    double u = clampf(u_unsat, out_min, out_max);

    if (dt_valid) {
        const bool saturated_high = u >= out_max && error > 0.0;
        const bool saturated_low = u <= out_min && error < 0.0;
        if (!(saturated_high || saturated_low)) {
            integral_ += error * dt;
            integral_ = clampf(integral_, i_min, i_max);
            // re-evaluate with the new integral
            u_unsat = p + ki * integral_ + d;
            u = clampf(u_unsat, out_min, out_max);
        }
    }
    return u;
}

// ============================================================== pure pursuit

PurePursuit::PurePursuit(double wheelbase_m, double lookahead_gain_)
    : wheelbase(wheelbase_m), lookahead_gain(lookahead_gain_) {}

double PurePursuit::steer_angle(const std::vector<double>& px,
                                const std::vector<double>& py,
                                const VehicleState& ego,
                                double lookahead_m) const {
    // Effective lookahead grows mildly with speed to damp oscillation.
    const double ld = std::max(lookahead_m, lookahead_gain * ego.speed + 2.0);

    // First point at/eyond the lookahead circle; else nearest.
    std::size_t i_goal = px.size();  // sentinel: none found
    for (std::size_t i = 0; i < px.size(); ++i) {
        const double dx = px[i] - ego.x, dy = py[i] - ego.y;
        if (dx * dx + dy * dy >= ld * ld) {
            i_goal = i;
            break;
        }
    }
    if (i_goal == px.size()) i_goal = argmin_d2(px, py, ego.x, ego.y);

    // bearing to goal in the ego frame
    const double gx = px[i_goal] - ego.x, gy = py[i_goal] - ego.y;
    const double cy = std::cos(-ego.yaw), sy = std::sin(-ego.yaw);
    const double xl = gx * cy - gy * sy;
    const double yl = gx * sy + gy * cy;
    const double alpha = std::atan2(yl, xl);

    return std::atan2(2.0 * wheelbase * std::sin(alpha), ld);
}

double PurePursuit::steer(const Trajectory& traj, const VehicleState& ego,
                          double lookahead_m, double max_steer_rad) const {
    if (traj.points.empty()) return 0.0;
    std::vector<double> px, py;
    px.reserve(traj.points.size());
    py.reserve(traj.points.size());
    for (const Waypoint& p : traj.points) {
        px.push_back(p.x);
        py.push_back(p.y);
    }
    const double delta = steer_angle(px, py, ego, lookahead_m);
    return clampf(delta / max_steer_rad, -1.0, 1.0);
}

// ================================================================== stanley

StanleyLateral::StanleyLateral(double wheelbase_m, double max_steer_deg,
                               double k_stanley_, double k_soft_mps)
    : wheelbase(wheelbase_m),
      max_steer_rad(max_steer_deg * kPi / 180.0),
      k_stanley(k_stanley_),
      k_soft(k_soft_mps) {}

double StanleyLateral::steer_rad(double theta_path, double ego_yaw, double e_fa,
                                 double speed) const {
    const double psi = wrap_angle(theta_path - ego_yaw);
    return psi - std::atan2(k_stanley * e_fa, speed + k_soft);
}

double StanleyLateral::steer(const Trajectory& traj,
                             const VehicleState& ego) const {
    if (traj.points.size() < 2) return 0.0;
    std::vector<double> px, py, headings;
    path_geometry(traj, px, py, headings);

    // Front-axle position (Stanley is defined w.r.t. the front axle).
    const double fx = ego.x + 0.5 * wheelbase * std::cos(ego.yaw);
    const double fy = ego.y + 0.5 * wheelbase * std::sin(ego.yaw);
    const std::size_t i_near = argmin_d2(px, py, fx, fy);

    const double theta_p = headings[i_near];
    // signed cross-track error: + means vehicle left of the path
    const double nx = -std::sin(theta_p), ny = std::cos(theta_p);  // left normal
    const double e_fa = (fx - px[i_near]) * nx + (fy - py[i_near]) * ny;
    const double delta = steer_rad(theta_p, ego.yaw, e_fa, ego.speed);
    return clampf(delta / max_steer_rad, -1.0, 1.0);
}

// ===================================================== lateral (composite)

LateralController::LateralController(double wheelbase_m, double max_steer_deg,
                                     double k_stanley_, double k_soft_mps,
                                     double low_speed_mps,
                                     double lookahead_gain_)
    : wheelbase(wheelbase_m),
      max_steer_rad(max_steer_deg * kPi / 180.0),
      k_stanley(k_stanley_),
      k_soft(k_soft_mps),
      low_speed(low_speed_mps),
      lookahead_gain(lookahead_gain_),
      stanley_(wheelbase_m, max_steer_deg, k_stanley_, k_soft_mps),
      pursuit_(wheelbase_m, lookahead_gain_) {}

double LateralController::steer(const Trajectory& traj, const VehicleState& ego,
                                double lookahead_m) const {
    if (traj.points.size() < 2) return 0.0;
    std::vector<double> px, py, headings;
    path_geometry(traj, px, py, headings);

    double delta;
    if (ego.speed < low_speed) {
        // At crawl speeds the Stanley cross-track term saturates and heading
        // estimates are noisy — pure pursuit is the better tracker.
        delta = pursuit_.steer_angle(px, py, ego, lookahead_m);
    } else {
        // Front-axle position (Stanley is defined w.r.t. the front axle).
        const double fx = ego.x + 0.5 * wheelbase * std::cos(ego.yaw);
        const double fy = ego.y + 0.5 * wheelbase * std::sin(ego.yaw);
        const std::size_t i_near = argmin_d2(px, py, fx, fy);

        const double theta_p = headings[i_near];
        const double nx = -std::sin(theta_p), ny = std::cos(theta_p);
        const double e_fa = (fx - px[i_near]) * nx + (fy - py[i_near]) * ny;
        delta = stanley_.steer_rad(theta_p, ego.yaw, e_fa, ego.speed);
    }
    return clampf(delta / max_steer_rad, -1.0, 1.0);
}

// ============================================================= longitudinal

LongitudinalController::LongitudinalController(
    double kp, double ki, double kd, double max_accel_mps2,
    double max_brake_mps2, double v_max_mps, double accel_deadband,
    double stop_margin_m, double min_ttc_s, double brake_hold_,
    double ego_half_length_m)
    : max_accel(max_accel_mps2),
      max_brake(max_brake_mps2),
      v_max(v_max_mps),
      deadband(accel_deadband),
      stop_margin(stop_margin_m),
      min_ttc(min_ttc_s),
      brake_hold(brake_hold_),
      ego_half_length(ego_half_length_m),
      pid(kp, ki, kd,
          -max_brake_mps2 / std::max(ki, 1e-6),
          max_accel_mps2 / std::max(ki, 1e-6),
          -max_brake_mps2, max_accel_mps2, 0.05, "longitudinal") {}

ControlCommand LongitudinalController::accel_cmd(
    double target_speed, const VehicleState& ego,
    const PerceptionOutput* perception) {
    const double dt = dt_from(ego.timestamp);
    double v_cmd = std::max(target_speed, 0.0);

    double forced_brake = 0.0;
    if (perception) {
        // ---- free-space stopping cap ---------------------------------- //
        const double d_eff = perception->free_space_ahead - stop_margin;
        const double v_cap =
            std::sqrt(std::max(0.0, 2.0 * 0.7 * max_brake * d_eff));
        v_cmd = std::min(v_cmd, v_cap);

        // ---- TTC guard against the nearest object ahead ---------------- //
        const auto [ttc_s, gap_m, closing_mps] = ttc(*perception, ego);
        (void)gap_m;
        (void)closing_mps;
        if (ttc_s < min_ttc) {
            v_cmd = 0.0;
            // harder brake as TTC shrinks toward zero
            forced_brake =
                std::min(1.0, 0.4 + 0.6 * (1.0 - ttc_s / min_ttc));
        }
    }

    // ---- PID on speed error ------------------------------------------- //
    const double err = v_cmd - ego.speed;
    const double a_des = pid.step(err, dt);

    ControlCommand cmd = accel_to_command(a_des, ego.speed);
    if (forced_brake > 0.0) {
        cmd.throttle = 0.0;
        cmd.brake = std::max(cmd.brake, forced_brake);
    }

    // hold a stopped car in place instead of rolling
    if (v_cmd < 0.3 && ego.speed < 0.4) {
        cmd.throttle = 0.0;
        cmd.brake = std::max(cmd.brake, brake_hold);
    }
    return cmd.clamp();
}

ControlCommand LongitudinalController::accel_to_command(double a_des,
                                                        double speed) const {
    ControlCommand cmd;
    if (a_des > deadband) {
        // torque taper: available accel falls off as v -> v_max
        const double taper = std::max(0.25, 1.0 - std::max(speed, 0.0) / v_max);
        const double a_avail = max_accel * taper;
        cmd.throttle = a_des / std::max(a_avail, 0.1);
    } else if (a_des < -deadband) {
        cmd.brake = -a_des / max_brake;
    }
    // else: deadband -> coast (both 0)
    return cmd.clamp();
}

std::tuple<double, double, double> LongitudinalController::ttc(
    const PerceptionOutput& perception, const VehicleState& ego) const {
    const double cy = std::cos(ego.yaw), sy = std::sin(ego.yaw);
    double best_ttc = kInf, best_gap = kInf, best_closing = 0.0;
    for (const DetectedObject& o : perception.objects) {
        const double dx = o.position.x - ego.x;
        const double dy = o.position.y - ego.y;
        const double lon = dx * cy + dy * sy;
        const double lat = -dx * sy + dy * cy;
        if (lon <= 0.0 || std::abs(lat) > 2.0) continue;
        const double gap =
            std::max(lon - o.bbox_extent.x - ego_half_length, 0.05);
        const double closing = ego.speed - o.velocity.norm();
        const double t = closing > 0.05 ? gap / closing : kInf;
        if (t < best_ttc) {
            best_ttc = t;
            best_gap = gap;
            best_closing = closing;
        }
    }
    return {best_ttc, best_gap, best_closing};
}

double LongitudinalController::dt_from(double ts) {
    if (!last_ts_) {
        last_ts_ = ts;
        return 0.02;
    }
    const double dt = ts - *last_ts_;
    last_ts_ = ts;
    return std::min(std::max(dt, 1e-3), 0.25);
}

// ================================================================= MPCLite

MPCLite::MPCLite(double wheelbase_m, double max_steer_deg, double horizon_s_,
                 double dt_, double v_max_mps, double max_accel_mps2,
                 double max_brake_mps2, double w_cte_, double w_yaw_,
                 double w_vel_, double w_effort_, double w_terminal_,
                 int refine_passes_)
    : L(wheelbase_m),
      max_steer(max_steer_deg * kPi / 180.0),
      horizon_s(horizon_s_),
      dt(dt_),
      steps(std::max(static_cast<int>(std::lround(horizon_s_ / dt_)), 1)),
      v_max(v_max_mps),
      max_accel(max_accel_mps2),
      max_brake(max_brake_mps2),
      w_cte(w_cte_),
      w_yaw(w_yaw_),
      w_vel(w_vel_),
      w_effort(w_effort_),
      w_terminal(w_terminal_),
      refine_passes(refine_passes_) {}

MPCResult MPCLite::optimize(const Trajectory& traj, const VehicleState& ego,
                            double target_speed, double base_steer_norm,
                            double base_accel) const {
    if (traj.points.size() < 2) return {base_steer_norm, base_accel};

    // reference arrays
    std::vector<double> rx, ry, rhead;
    path_geometry(traj, rx, ry, rhead);

    // ---- stage 1: coarse absolute grid --------------------------------- //
    const std::vector<double> steers = {-1.0 * max_steer, -0.5 * max_steer, 0.0,
                                        0.5 * max_steer, 1.0 * max_steer};
    const std::vector<double> accels = {-max_brake, -0.5 * max_brake, 0.0,
                                        0.6 * max_accel, max_accel};
    auto best = search(rx, ry, rhead, ego, target_speed, steers, accels);
    if (!best) return {base_steer_norm, base_accel};

    // ---- stage 2: local refinement around the winner -------------------- //
    for (int pass = 0; pass < refine_passes; ++pass) {
        const auto [s0, a0] = *best;
        std::vector<double> s_fine(5), a_fine(5);
        const double s_off[5] = {-0.25, -0.125, 0.0, 0.125, 0.25};
        const double a_off[5] = {-0.5, -0.25, 0.0, 0.25, 0.5};
        for (int i = 0; i < 5; ++i) {
            s_fine[i] = clampf(s0 + s_off[i] * max_steer, -max_steer, max_steer);
            a_fine[i] = clampf(a0 + a_off[i] * max_accel, -max_brake, max_accel);
        }
        auto fine = search(rx, ry, rhead, ego, target_speed, s_fine, a_fine);
        if (fine) best = fine;
    }

    MPCResult out;
    out.steer_norm = clampf(best->first / max_steer, -1.0, 1.0);
    out.accel = clampf(best->second, -max_brake, max_accel);
    return out;
}

std::optional<std::pair<double, double>> MPCLite::search(
    const std::vector<double>& rx, const std::vector<double>& ry,
    const std::vector<double>& rhead, const VehicleState& ego, double v_ref,
    const std::vector<double>& steers,
    const std::vector<double>& accels) const {
    double best_cost = kInf;
    std::optional<std::pair<double, double>> best;
    for (double delta : steers) {
        const double tan_delta_over_l = std::tan(delta) / L;
        for (double accel : accels) {
            const double cost = rollout_cost(rx, ry, rhead, ego, v_ref, delta,
                                             tan_delta_over_l, accel);
            if (cost < best_cost) {
                best_cost = cost;
                best = {delta, accel};
            }
        }
    }
    return best;
}

double MPCLite::rollout_cost(const std::vector<double>& rx,
                             const std::vector<double>& ry,
                             const std::vector<double>& rhead,
                             const VehicleState& ego, double v_ref,
                             double delta, double tan_delta_over_l,
                             double accel) const {
    double x = ego.x, y = ego.y, yaw = ego.yaw, v = std::max(ego.speed, 0.0);
    double cost = 0.0;
    double cos_y = std::cos(yaw), sin_y = std::sin(yaw);

    for (int k = 0; k < steps; ++k) {
        // kinematic bicycle update
        x += v * cos_y * dt;
        y += v * sin_y * dt;
        yaw += v * tan_delta_over_l * dt;
        v = clampf(v + accel * dt, 0.0, v_max);
        cos_y = std::cos(yaw);
        sin_y = std::sin(yaw);

        // nearest reference point -> cross-track + heading error
        const std::size_t i = argmin_d2(rx, ry, x, y);
        const double th = rhead[i];
        const double cte = (x - rx[i]) * -std::sin(th) + (y - ry[i]) * std::cos(th);
        const double yaw_err = wrap_angle(th - yaw);
        const double vel_err = v - v_ref;

        const double w = (k == steps - 1) ? w_terminal : 1.0;
        cost += w * (w_cte * cte * cte + w_yaw * yaw_err * yaw_err +
                     w_vel * vel_err * vel_err);
    }

    // control effort, normalized
    const double a_norm =
        accel < 0.0 ? accel / max_brake : accel / max_accel;
    cost += w_effort * ((delta / max_steer) * (delta / max_steer) + a_norm * a_norm);
    return cost;
}

// ============================================================ facade

VehicleController::VehicleController(
    double wheelbase_m, double max_steer_deg, double max_steer_rate_,
    double max_accel_mps2, double max_brake_mps2, double lookahead_m_,
    bool use_mpc_, double mpc_blend_, double mpc_min_speed_)
    : lateral(wheelbase_m, max_steer_deg),
      longitudinal(1.2, 0.25, 0.05, max_accel_mps2, max_brake_mps2),
      mpc(wheelbase_m, max_steer_deg, 1.2, 0.1, 16.7, max_accel_mps2,
          max_brake_mps2),
      lookahead_m(lookahead_m_),
      max_steer_rate(max_steer_rate_),
      use_mpc(use_mpc_),
      mpc_blend(clampf(mpc_blend_, 0.0, 1.0)),
      mpc_min_speed(mpc_min_speed_),
      max_accel(max_accel_mps2),
      max_brake(max_brake_mps2) {}

ControlCommand VehicleController::compute(const Trajectory* trajectory,
                                          const VehicleState& ego,
                                          const PerceptionOutput* perception) {
    const double dt = dt_from(ego.timestamp);

    // Degenerate input -> controlled stop, unwind the wheel gently.
    if (trajectory == nullptr || trajectory->empty()) {
        ControlCommand c;
        c.steer = rate_limit(0.0, dt);
        c.brake = 0.8;
        return c.clamp();
    }

    // ---- lateral ------------------------------------------------------ //
    double steer = lateral.steer(*trajectory, ego, lookahead_m);

    // ---- longitudinal ------------------------------------------------- //
    const double target = std::max(trajectory->target_speed, 0.0);
    ControlCommand long_cmd = longitudinal.accel_cmd(target, ego, perception);
    const double a_base =
        long_cmd.throttle * max_accel - long_cmd.brake * max_brake;

    // ---- MPC-lite refinement ------------------------------------------ //
    if (use_mpc && ego.speed >= mpc_min_speed) {
        const MPCResult res =
            mpc.optimize(*trajectory, ego, target, steer, a_base);
        const double w = mpc_blend;
        steer = (1.0 - w) * steer + w * res.steer_norm;
        const double a_ref = (1.0 - w) * a_base + w * res.accel;
        ControlCommand blended = longitudinal.accel_to_command(a_ref, ego.speed);
        // Never let the optimizer weaken a safety brake, and never let it add
        // throttle while the PID is braking or holding a stop.
        blended.brake = std::max(blended.brake, long_cmd.brake);
        if (long_cmd.brake > 0.0 || target < 0.3) {
            blended.throttle = 0.0;
        } else {
            blended.throttle = std::min(
                blended.throttle, std::max(long_cmd.throttle * 1.5, 0.2));
        }
        long_cmd = blended;
    }

    steer = rate_limit(steer, dt);

    ControlCommand c;
    c.throttle = long_cmd.throttle;
    c.brake = long_cmd.brake;
    c.steer = steer;
    return c.clamp();
}

void VehicleController::reset() {
    prev_steer_ = 0.0;
    prev_ts_ = std::nullopt;
    longitudinal.pid.reset();
}

double VehicleController::rate_limit(double steer, double dt) {
    const double max_d = max_steer_rate * dt;
    const double limited =
        clampf(steer, prev_steer_ - max_d, prev_steer_ + max_d);
    prev_steer_ = clampf(limited, -1.0, 1.0);
    return prev_steer_;
}

double VehicleController::dt_from(double ts) {
    if (!prev_ts_) {
        prev_ts_ = ts;
        return 0.02;
    }
    const double dt = ts - *prev_ts_;
    prev_ts_ = ts;
    return std::min(std::max(dt, 1e-3), 0.25);
}

}  // namespace fsd
