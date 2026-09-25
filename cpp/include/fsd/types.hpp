// Shared data contracts for the FSD C++ core — header-only.
//
// Mirrors fsd/core/types.py field-for-field so the Python stack can treat the
// pybind11 bindings as a drop-in backend. Every module talks in these types.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace fsd {

constexpr double kPi = 3.14159265358979323846264338327950288;
constexpr double kInf = std::numeric_limits<double>::infinity();

/// Wall-clock seconds since epoch — the same domain as Python's time.time().
/// Timestamps in these contracts live in this domain; inject ClockFn where
/// determinism is required (the safety monitor and smoke test do this).
inline double now_seconds() {
    return std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch())
        .count();
}

/// Injectable time source. Returning a synthetic tick counter makes every
/// staleness/watchdog path fully deterministic.
using ClockFn = std::function<double()>;

// --------------------------------------------------------------------- enums

enum class LaneChangeState { KEEP, LEFT, RIGHT };
enum class LightState { RED, YELLOW, GREEN, UNKNOWN };

/// Arbitration output of the safety gate. DISENGAGED is sticky: only an
/// explicit reset() leaves it (mirrors the Python monitor).
enum class DriveMode { ENGAGED, DEGRADED, SAFE_STOP, DISENGAGED };

/// Names match the Python Enum.name values so logs/status dicts line up.
inline const char* to_string(DriveMode m) {
    switch (m) {
        case DriveMode::ENGAGED: return "ENGAGED";
        case DriveMode::DEGRADED: return "DEGRADED";
        case DriveMode::SAFE_STOP: return "SAFE_STOP";
        case DriveMode::DISENGAGED: return "DISENGAGED";
    }
    return "UNKNOWN";
}
inline const char* to_string(LightState s) {
    switch (s) {
        case LightState::RED: return "RED";
        case LightState::YELLOW: return "YELLOW";
        case LightState::GREEN: return "GREEN";
        case LightState::UNKNOWN: return "UNKNOWN";
    }
    return "UNKNOWN";
}
inline const char* to_string(LaneChangeState s) {
    switch (s) {
        case LaneChangeState::KEEP: return "KEEP";
        case LaneChangeState::LEFT: return "LEFT";
        case LaneChangeState::RIGHT: return "RIGHT";
    }
    return "UNKNOWN";
}

// ------------------------------------------------------------------ helpers

inline double clampf(double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}
inline bool finite(double x) { return std::isfinite(x); }
inline bool all_finite(std::initializer_list<double> xs) {
    for (double x : xs)
        if (!std::isfinite(x)) return false;
    return true;
}
/// Wrap to [-pi, pi), matching Python's `(a + pi) % (2*pi) - pi`.
inline double wrap_angle(double a) {
    double w = std::fmod(a + kPi, 2.0 * kPi);
    if (w < 0.0) w += 2.0 * kPi;
    return w - kPi;
}
/// Python's round() / int(round(x)): round-half-even (IEEE default mode).
inline long round_nearest(double x) { return std::lrint(x); }

/// Severity ordering used when arbitrating between rule events.
/// "info"=0, "warning"=1, "critical"=2; unknown levels count as info.
inline int severity_order(const std::string& level) {
    if (level == "critical") return 2;
    if (level == "warning") return 1;
    return 0;
}

/// DriveMode escalation priority — DISENGAGED highest (sticky).
inline int mode_priority(DriveMode m) {
    switch (m) {
        case DriveMode::ENGAGED: return 0;
        case DriveMode::DEGRADED: return 1;
        case DriveMode::SAFE_STOP: return 2;
        case DriveMode::DISENGAGED: return 3;
    }
    return 0;
}

// ------------------------------------------------------------------ structs

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double norm() const { return std::sqrt(x * x + y * y + z * z); }
};

/// Ego-vehicle kinematic state, world frame.
struct VehicleState {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double yaw = 0.0;    ///< radians
    double speed = 0.0;  ///< m/s
    double accel = 0.0;  ///< m/s^2
    double steer = 0.0;  ///< normalized [-1, 1]
    double timestamp = now_seconds();
};

struct Waypoint {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double yaw = 0.0;
    double speed_limit = 13.9;  ///< m/s, ~50 km/h default
};

struct Trajectory {
    std::vector<Waypoint> points;
    double target_speed = 0.0;
    double horizon_s = 4.0;
    bool empty() const { return points.empty(); }
};

/// A fused perception object.
struct DetectedObject {
    int64_t obj_id = 0;
    std::string cls;  ///< vehicle | pedestrian | cyclist | sign | misc
    Vec3 position;
    Vec3 velocity;
    Vec3 bbox_extent;
    double confidence = 0.0;
    double timestamp = now_seconds();
};

struct LaneInfo {
    double left_offset = 0.0;   ///< m, + = left lane edge to the left of ego
    double right_offset = 0.0;
    double center_offset = 0.0; ///< m, signed lateral error from lane center
    double heading_error = 0.0; ///< rad
    double curvature = 0.0;     ///< 1/m
    double lane_width = 3.5;
    bool detected = true;
};

/// Actuation demand sent to the vehicle.
struct ControlCommand {
    double throttle = 0.0;  ///< [0, 1]
    double brake = 0.0;     ///< [0, 1]
    double steer = 0.0;     ///< [-1, 1]
    bool hand_brake = false;
    bool reverse = false;

    ControlCommand& clamp() {
        throttle = clampf(throttle, 0.0, 1.0);
        brake = clampf(brake, 0.0, 1.0);
        steer = clampf(steer, -1.0, 1.0);
        return *this;
    }
};

struct SafetyEvent {
    std::string level = "info";  ///< info | warning | critical
    std::string source;
    std::string message;
    double timestamp = now_seconds();

    SafetyEvent() = default;
    SafetyEvent(std::string lvl, std::string src, std::string msg)
        : level(std::move(lvl)), source(std::move(src)), message(std::move(msg)) {}
};

struct PerceptionOutput {
    std::vector<DetectedObject> objects;
    std::optional<LaneInfo> lane;              ///< nullopt == no lane estimate
    LightState light = LightState::UNKNOWN;
    double free_space_ahead = 100.0;           ///< m of clear path ahead
    double timestamp = now_seconds();
};

/// Mirror of fsd.core.config.SafetyConfig — the safety envelope the rules use.
struct SafetyConfig {
    double min_ttc_s = 1.5;          ///< time-to-collision floor
    double max_speed_mps = 16.7;     ///< 60 km/h cap
    double max_accel_mps2 = 3.0;
    double max_brake_mps2 = 6.0;
    double watchdog_timeout_s = 0.5; ///< pipeline stall -> SAFE_STOP
    double min_free_space_m = 8.0;
    double max_steer_rate = 0.4;     ///< normalized units/s
};

}  // namespace fsd
