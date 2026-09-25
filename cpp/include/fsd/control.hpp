// Control stack — C++17 port of fsd/control/{pid,lateral,longitudinal,mpc,controller}.py
//
//   * PIDController          — scalar PID, conditional anti-windup, filtered D.
//   * PurePursuit            — geometric lookahead tracker (low-speed fallback).
//   * StanleyLateral         — Stanley (2005) path-tracking law, front axle.
//   * LateralController      — Stanley + pure-pursuit fallback, normalized out.
//   * LongitudinalController — speed PID -> throttle/brake split, TTC guard.
//   * MPCLite                — grid-search shooting MPC over kinematic bicycle.
//   * VehicleController      — facade: Trajectory + state + perception -> cmd.
#pragma once

#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "fsd/types.hpp"

namespace fsd {

/// Scalar PID with clamping, conditional anti-windup and filtered D-term.
///
///     u[k] = Kp*e[k] + Ki * sum(e*dt) + Kd * d(e)/dt
///
/// Anti-windup: conditional integration — the integrator freezes whenever the
/// output is saturated *and* the error would push it deeper into saturation.
/// The integral term is additionally hard-clamped to [i_min, i_max]. The
/// derivative channel is low-pass filtered (first-order, tau = deriv_tau).
class PIDController {
public:
    double kp;
    double ki;
    double kd;
    double i_min;
    double i_max;
    double out_min;
    double out_max;
    double deriv_tau;
    std::string name;

    explicit PIDController(double kp = 1.0, double ki = 0.0, double kd = 0.0,
                           double i_min = -1.0, double i_max = 1.0,
                           double out_min = -kInf, double out_max = kInf,
                           double deriv_tau = 0.05, std::string name = "pid");

    void reset();
    double integral() const { return integral_; }

    /// Advance one tick. `dt` is seconds since the previous call.
    /// A non-positive or absurdly large dt is treated as a stalled tick:
    /// proportional action still applies, integration/derivative are skipped.
    double step(double error, double dt);
    double operator()(double error, double dt) { return step(error, dt); }

private:
    double integral_ = 0.0;
    std::optional<double> prev_error_;
    double deriv_ = 0.0;
};

/// Geometric pure pursuit — the low-speed fallback.
///
///     delta = atan2(2 * L * sin(alpha), l_d)
///
/// with alpha the bearing to the lookahead point in the ego frame.
class PurePursuit {
public:
    double wheelbase;       ///< m
    double lookahead_gain;  ///< effective lookahead grows mildly with speed

    explicit PurePursuit(double wheelbase_m = 2.875, double lookahead_gain = 0.35);

    /// Steer angle (radians) toward the lookahead point on the path.
    /// px/py are the path point coordinates (same length, >= 1).
    double steer_angle(const std::vector<double>& px,
                       const std::vector<double>& py, const VehicleState& ego,
                       double lookahead_m) const;

    /// Convenience: normalized steer in [-1, 1] for a Trajectory.
    double steer(const Trajectory& traj, const VehicleState& ego,
                 double lookahead_m, double max_steer_rad) const;
};

/// Stanley path-tracking law (Stanford, 2005):
///
///     delta = psi - atan2(k * e_fa, v + k_soft)
///
/// psi = heading error wrap(theta_path - theta_ego); e_fa is the signed
/// cross-track error of the *front axle* (+ = vehicle left of the path).
/// Positive steer = left. If the vehicle is left of the path (e_fa > 0) the
/// atan term commands a right turn — hence the minus.
class StanleyLateral {
public:
    double wheelbase;
    double max_steer_rad;
    double k_stanley;
    double k_soft;

    explicit StanleyLateral(double wheelbase_m = 2.875, double max_steer_deg = 60.0,
                            double k_stanley = 2.5, double k_soft_mps = 1.2);

    /// The Stanley law itself — steer angle in radians.
    double steer_rad(double theta_path, double ego_yaw, double e_fa,
                     double speed) const;

    /// Normalized steer in [-1, 1] against the nearest path point to the
    /// front axle. Returns 0 for empty/degenerate trajectories.
    double steer(const Trajectory& traj, const VehicleState& ego) const;
};

/// Stanley controller with pure-pursuit fallback for low speed
/// (the Python LateralController; composed here from the two pieces above).
class LateralController {
public:
    double wheelbase;
    double max_steer_rad;
    double k_stanley;
    double k_soft;
    double low_speed;
    double lookahead_gain;

    explicit LateralController(double wheelbase_m = 2.875,
                               double max_steer_deg = 60.0,
                               double k_stanley = 2.5, double k_soft_mps = 1.2,
                               double low_speed_mps = 0.6,
                               double lookahead_gain = 0.35);

    /// Return normalized steer in [-1, 1]. Positive = left.
    double steer(const Trajectory& traj, const VehicleState& ego,
                 double lookahead_m = 8.0) const;

private:
    StanleyLateral stanley_;
    PurePursuit pursuit_;
};

/// Speed-tracking PID with throttle/brake split and TTC guard.
///
/// Pipeline per tick:
///   1. Safety cap on requested speed — never command a speed from which the
///      vehicle could not stop within free_space_ahead using 70% of max brake.
///   2. TTC guard — below min_ttc_s to the nearest corridor object, zero the
///      target and force brake.
///   3. PID on speed error -> desired acceleration a_des.
///   4. accel -> actuator split with a speed-dependent torque taper.
class LongitudinalController {
public:
    double max_accel;
    double max_brake;
    double v_max;
    double deadband;
    double stop_margin;
    double min_ttc;
    double brake_hold;
    double ego_half_length;

    PIDController pid;

    explicit LongitudinalController(
        double kp = 1.2, double ki = 0.25, double kd = 0.05,
        double max_accel_mps2 = 3.0, double max_brake_mps2 = 6.0,
        double v_max_mps = 55.0, double accel_deadband = 0.08,
        double stop_margin_m = 2.0, double min_ttc_s = 1.2,
        double brake_hold = 0.35, double ego_half_length_m = 2.4);

    /// Track `target_speed`; returns a throttle/brake-only command (steer=0).
    /// `perception` may be nullptr (no guarding applied).
    ControlCommand accel_cmd(double target_speed, const VehicleState& ego,
                             const PerceptionOutput* perception = nullptr);

    /// Split a desired acceleration into throttle/brake in [0, 1].
    ControlCommand accel_to_command(double a_des, double speed) const;

private:
    /// Time-to-collision with the nearest object in our corridor.
    /// Returns (ttc_s, gap_m, closing_mps); ttc inf when nothing threatens.
    std::tuple<double, double, double> ttc(const PerceptionOutput& perception,
                                           const VehicleState& ego) const;
    double dt_from(double ts);

    std::optional<double> last_ts_;
};

/// Lightweight shooting MPC — a candidate-control ("brute-force shooting")
/// approximation over the kinematic bicycle model, NOT a QP solver.
///
/// The control space is a small (steer, accel) grid held constant over a
/// ~1.2 s horizon; rollouts are scored on cross-track, heading, speed error
/// and control effort; a second finer pass refines the stage-1 winner.
/// The returned pair is the *first* control of the best rollout.
struct MPCResult {
    double steer_norm = 0.0;  ///< normalized [-1, 1]
    double accel = 0.0;       ///< m/s^2
};

class MPCLite {
public:
    double L;          ///< wheelbase, m
    double max_steer;  ///< rad
    double horizon_s;
    double dt;
    int steps;
    double v_max;
    double max_accel;
    double max_brake;
    double w_cte, w_yaw, w_vel, w_effort, w_terminal;
    int refine_passes;

    explicit MPCLite(double wheelbase_m = 2.875, double max_steer_deg = 60.0,
                     double horizon_s = 1.2, double dt = 0.1,
                     double v_max_mps = 16.7, double max_accel_mps2 = 3.0,
                     double max_brake_mps2 = 6.0, double w_cte = 6.0,
                     double w_yaw = 1.5, double w_vel = 0.4,
                     double w_effort = 0.15, double w_terminal = 2.0,
                     int refine_passes = 1);

    /// Return (steer_normalized, accel_mps2) for the current tick.
    MPCResult optimize(const Trajectory& traj, const VehicleState& ego,
                       double target_speed, double base_steer_norm = 0.0,
                       double base_accel = 0.0) const;

private:
    std::optional<std::pair<double, double>> search(
        const std::vector<double>& rx, const std::vector<double>& ry,
        const std::vector<double>& rhead, const VehicleState& ego, double v_ref,
        const std::vector<double>& steers,
        const std::vector<double>& accels) const;

    double rollout_cost(const std::vector<double>& rx,
                        const std::vector<double>& ry,
                        const std::vector<double>& rhead,
                        const VehicleState& ego, double v_ref, double delta,
                        double tan_delta_over_l, double accel) const;
};

/// Top-level controller: Trajectory + state + perception -> command.
///
/// Per tick:
///     steer_base = Stanley(trajectory, ego)   [pure pursuit at crawl speed]
///     accel_base = longitudinal PID -> throttle/brake split
///     (steer_mpc, accel_mpc) = MPCLite refinement, blended at mpc_blend
///
/// The MPC blend is deliberately modest: the analytic controllers stay
/// primary. Empty/missing trajectory -> firm brake, steer decays to straight.
/// Steer is rate-limited (normalized units/s) so the safety envelope's
/// max_steer_rate is honored even when planners jump the reference path.
class VehicleController {
public:
    LateralController lateral;
    LongitudinalController longitudinal;
    MPCLite mpc;

    double lookahead_m;
    double max_steer_rate;
    bool use_mpc;
    double mpc_blend;
    double mpc_min_speed;
    double max_accel;
    double max_brake;

    explicit VehicleController(
        double wheelbase_m = 2.875, double max_steer_deg = 60.0,
        double max_steer_rate = 0.4, double max_accel_mps2 = 3.0,
        double max_brake_mps2 = 6.0, double lookahead_m = 8.0,
        bool use_mpc = true, double mpc_blend = 0.35, double mpc_min_speed = 1.0);

    /// Produce the clamped actuation command for this tick.
    /// `trajectory`/`perception` may be nullptr (None in the Python API).
    ControlCommand compute(const Trajectory* trajectory, const VehicleState& ego,
                           const PerceptionOutput* perception = nullptr);
    /// Friendly alias for tick-style callers.
    ControlCommand step(const Trajectory* trajectory, const VehicleState& ego,
                        const PerceptionOutput* perception = nullptr) {
        return compute(trajectory, ego, perception);
    }

    void reset();

private:
    double rate_limit(double steer, double dt);
    double dt_from(double ts);

    double prev_steer_ = 0.0;
    std::optional<double> prev_ts_;
};

}  // namespace fsd
