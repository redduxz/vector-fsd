// SafetyMonitor — the hard gate between the planner and the actuators.
//
// C++17 port of fsd/safety/rules.py + fsd/safety/monitor.py. Same rule set,
// same arbitration semantics:
//
//   mode = monitor.check(ego, perception, demand_cmd, pipeline_alive);
//   if (mode == DriveMode::SAFE_STOP)
//       actuate(monitor.engage_safe_stop());
//   else
//       actuate(monitor.enforce(demand_cmd));
//
// Design principles (unchanged from Python):
//   * Fail-safe everywhere — internal errors, malformed input, stale sensors
//     or faulting rules degrade toward SAFE_STOP, never toward silence.
//   * Escalate instantly, de-escalate slowly — warnings hold for
//     warn_clear_s of clean cycles; criticals latch until reset().
//   * Enforcement is separate from detection — rules flag events, the monitor
//     arbitrates a DriveMode, then runs each rule's enforce() on its own copy
//     of the command so the planner's object is never mutated.
#pragma once

#include <cstddef>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "fsd/types.hpp"

namespace fsd {

/// Nominal actuator gains (fraction -> m/s^2) for a Model-3-class vehicle.
/// Used to translate unitless throttle/brake demands into accel estimates.
inline constexpr double kNominalMaxAccelMps2 = 6.0;  ///< ~full throttle at road speed
inline constexpr double kNominalMaxBrakeMps2 = 8.0;  ///< ~full brake, dry asphalt

inline constexpr double kEgoHalfLengthM = 2.4;  ///< front bumper to center + margin
inline constexpr double kEgoHalfWidthM = 1.0;   ///< mirror to center + margin

/// Optional log sink: (level, message). Silent unless installed.
using LogFn = std::function<void(const std::string& level, const std::string& message)>;

// --------------------------------------------------------------------- ctx

/// Everything a rule is allowed to look at for one evaluation cycle.
struct SafetyContext {
    VehicleState ego;
    std::optional<PerceptionOutput> perception;
    ControlCommand cmd;                        ///< planner demand — never mutate
    std::optional<ControlCommand> applied;     ///< last command the monitor let through
    bool pipeline_alive = true;                ///< upstream heartbeat flag from caller
    double now = 0.0;                          ///< clock() at start of check
    double dt = 0.05;                          ///< seconds since previous check
    SafetyConfig cfg;
    double last_heartbeat = 0.0;               ///< last explicit pipeline heartbeat
};

// -------------------------------------------------------------------- rules

/// Base class for a check. Subclasses override evaluate() and/or enforce().
/// Rules must never throw — but the monitor guards every call anyway, because
/// a faulting rule in the safety layer is itself a critical event.
class SafetyRule {
public:
    virtual ~SafetyRule() = default;

    std::string name = "rule";
    bool enabled = true;

    /// Detect a violation and classify its severity, or nullopt if clean.
    virtual std::optional<SafetyEvent> evaluate(const SafetyContext& ctx) const;
    /// Reshape a demanded command to stay inside the safe envelope.
    virtual ControlCommand enforce(ControlCommand cmd, const SafetyContext& ctx) const;
    /// Drop any state carried between cycles (called on monitor reset).
    virtual void reset();
};

/// Time-to-collision against fused objects in the ego corridor.
///
/// An object counts if its lateral offset fits inside a corridor the width of
/// the ego car plus the object's own half-width, and it is ahead of the front
/// bumper. TTC = gap / closing-speed, evaluated along the ego heading.
class TTCRule : public SafetyRule {
public:
    static constexpr double MIN_CLOSING_MPS = 0.1;  ///< below this the gap is stable
    static constexpr double WARN_FACTOR = 1.8;      ///< warn at 1.8x the critical floor

    TTCRule() { name = "ttc"; }
    std::optional<SafetyEvent> evaluate(const SafetyContext& ctx) const override;
};

/// Caps vehicle speed and commanded acceleration.
///
/// Evaluation flags overspeed; enforcement zeroes throttle at the cap, adds
/// compensating brake when already over, tapers throttle just below the cap,
/// and bounds throttle/brake fractions by the configured accel/decel budgets.
class SpeedLimitRule : public SafetyRule {
public:
    static constexpr double TAPER_BAND_MPS = 2.0;  ///< start tapering this far below cap
    static constexpr double HARD_FACTOR = 1.35;    ///< >35% over the cap is critical

    SpeedLimitRule() { name = "speed_limit"; }
    std::optional<SafetyEvent> evaluate(const SafetyContext& ctx) const override;
    ControlCommand enforce(ControlCommand cmd, const SafetyContext& ctx) const override;
};

/// Drivable-space check: the path ahead must be measurably clear.
///
/// A perception blackout is represented upstream as free_space_ahead == 0,
/// which this rule treats the same as a physically blocked path.
class FreeSpaceRule : public SafetyRule {
public:
    static constexpr double WARN_FACTOR = 1.75;

    FreeSpaceRule() { name = "free_space"; }
    std::optional<SafetyEvent> evaluate(const SafetyContext& ctx) const override;
};

/// Liveness: pipeline heartbeat, and staleness of the fused inputs.
///
/// Two independent stall detectors: the explicit pipeline_alive flag /
/// heartbeat timestamp the upstream loop must maintain, and the freshness of
/// the perception/ego timestamps themselves.
class WatchdogRule : public SafetyRule {
public:
    WatchdogRule() { name = "watchdog"; }
    std::optional<SafetyEvent> evaluate(const SafetyContext& ctx) const override;
};

/// Rate-limits the steering demand.
///
/// The limit is applied against the last *enforced* steer (falling back to the
/// reported ego steer), so a planner cannot accumulate its way around the cap
/// by ramping demand faster than we let it through.
class SteerRateRule : public SafetyRule {
public:
    SteerRateRule() { name = "steer_rate"; }
    std::optional<SafetyEvent> evaluate(const SafetyContext& ctx) const override;
    ControlCommand enforce(ControlCommand cmd, const SafetyContext& ctx) const override;

private:
    double baseline(const SafetyContext& ctx) const;
};

/// Lane-keeping guard.
///
/// Fully departing the lane is critical; drifting toward the edge is a warning
/// that degrades the drive mode. Loss of lane tracking at speed is a warning.
class LaneDepartureRule : public SafetyRule {
public:
    static constexpr double LOST_MIN_SPEED_MPS = 4.0;  ///< below this, loss is not actionable
    static constexpr double EXIT_MARGIN_M = 0.4;       ///< fully out = edge + this margin

    LaneDepartureRule() { name = "lane_departure"; }
    std::optional<SafetyEvent> evaluate(const SafetyContext& ctx) const override;
};

/// The standard rule set, in enforcement order.
///
/// Order matters for enforce(): caps and taper first, steering slew last so
/// the final steer value is what leaves the gate.
std::vector<std::shared_ptr<SafetyRule>> default_rules();

// ------------------------------------------------------------------ monitor

/// Snapshot returned by SafetyMonitor::status().
struct MonitorStatus {
    std::string mode;
    double mode_for_s = 0.0;
    bool latched = false;
    bool hazards = false;
    double heartbeat_age_s = 0.0;
    std::vector<std::pair<std::string, bool>> rules;  ///< (name, enabled)
    std::unordered_map<std::string, int> rule_faults;
    std::unordered_map<std::string, int> event_counts;
    std::size_t events_logged = 0;
};

class SafetyMonitor {
public:
    /// `rules` empty -> default_rules() (same as Python's `rules=None`).
    explicit SafetyMonitor(SafetyConfig cfg = SafetyConfig{},
                           std::vector<std::shared_ptr<SafetyRule>> rules = {},
                           std::size_t event_history = 4096);

    // Public knobs mirroring the Python attributes (mutated between cycles is
    // supported — each check() re-reads them).
    SafetyConfig cfg;
    std::vector<std::shared_ptr<SafetyRule>> rules;
    bool latch_critical = true;
    double warn_clear_s = 0.5;
    double critical_clear_s = 3.0;
    double default_dt = 0.05;
    bool hazards_requested = false;  ///< vehicle bridge should map to hazard lamps

    // ------------------------------------------------------------------ API

    /// Run one arbitration cycle. Never throws — internal faults fail safe.
    DriveMode check(std::optional<VehicleState> ego,
                    std::optional<PerceptionOutput> perception,
                    std::optional<ControlCommand> cmd,
                    bool pipeline_alive = true);

    /// Return a command that is legal to send right now.
    ///
    /// Applies input sanitation, per-rule shaping, and the current drive
    /// mode's override. Never mutates the caller's object.
    ControlCommand enforce(std::optional<ControlCommand> cmd = std::nullopt);

    /// The minimal-risk command: full brake, no throttle, hold steer.
    /// Also raises hazards_requested — the vehicle bridge should map that to
    /// the hazard lamps, since ControlCommand carries no light channel.
    ControlCommand engage_safe_stop();

    /// Upstream pipeline calls this every healthy tick.
    void heartbeat() { last_heartbeat_ = clock_(); }

    /// Hand control to the human. Sticky until reset().
    void disengage();

    /// Clear latches and return to ENGAGED. Records a fresh heartbeat.
    void reset();

    // ------------------------------------------------------------ rule admin
    void add_rule(std::shared_ptr<SafetyRule> rule) { rules.push_back(std::move(rule)); }
    bool remove_rule(const std::string& name);
    bool set_rule_enabled(const std::string& name, bool enabled);
    std::shared_ptr<SafetyRule> get_rule(const std::string& name) const;

    // -------------------------------------------------------------- introspect
    DriveMode mode() const { return mode_; }
    bool latched() const { return latched_; }
    /// The command the monitor last produced (post-enforcement).
    const std::optional<ControlCommand>& last_command() const { return applied_; }
    std::vector<SafetyEvent> event_log() const { return {events_.begin(), events_.end()}; }
    const std::unordered_map<std::string, int>& event_counts() const { return counts_; }
    std::vector<SafetyEvent> recent_events(std::size_t n = 20) const;
    bool healthy() const { return mode_ == DriveMode::ENGAGED; }
    MonitorStatus status() const;

    /// Inject a synthetic time source (deterministic tests / sims).
    void set_clock(ClockFn fn) { clock_ = std::move(fn); }
    /// Install a log sink: (level, message). Level is info|warning|error.
    void set_logger(LogFn fn) { logger_ = std::move(fn); }

private:
    // -------------------------------------------------------------- internals
    DriveMode evaluate_cycle(std::optional<VehicleState> ego,
                             std::optional<PerceptionOutput> perception,
                             std::optional<ControlCommand> cmd,
                             bool pipeline_alive);
    /// Guarantee a finite VehicleState; substitute a parked car if needed.
    static VehicleState preflight_ego(std::optional<VehicleState> ego, double now,
                                      std::vector<SafetyEvent>& events);
    ControlCommand sanitize_cmd(ControlCommand c);
    /// What the monitor assumes when perception is absent: worst case.
    static PerceptionOutput blind_perception(double now);
    static DriveMode arbitrate(const std::vector<SafetyEvent>& events);
    void update_mode(DriveMode raw, double now);
    void log_event(const SafetyEvent& ev);
    void log(const std::string& level, const std::string& message) const;

    ClockFn clock_ = now_seconds;
    LogFn logger_;

    DriveMode mode_ = DriveMode::ENGAGED;
    double mode_since_ = 0.0;
    bool disengaged_ = false;
    bool latched_ = false;

    std::optional<double> last_check_;
    double last_heartbeat_ = 0.0;
    std::optional<SafetyContext> last_ctx_;
    std::optional<ControlCommand> applied_;
    double last_warn_t_ = -kInf;
    double last_critical_t_ = -kInf;
    bool safe_stop_announced_ = false;

    std::deque<SafetyEvent> events_;
    std::size_t event_history_ = 4096;
    std::unordered_map<std::string, int> counts_;
    std::unordered_map<std::string, int> rule_faults_;
};

}  // namespace fsd
