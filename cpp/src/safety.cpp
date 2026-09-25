// Implementation of the safety layer — a faithful C++17 port of
// fsd/safety/rules.py and fsd/safety/monitor.py.
#include "fsd/safety.hpp"

#include <cstdarg>
#include <cstdio>
#include <exception>

namespace fsd {
namespace {

/// printf-style formatting for event messages (heap-free fast path).
std::string fmt(const char* format, ...) {
    char buf[1024];
    std::va_list args;
    va_start(args, format);
    const int n = std::vsnprintf(buf, sizeof(buf), format, args);
    va_end(args);
    if (n < 0) return {};
    if (static_cast<std::size_t>(n) < sizeof(buf)) return std::string(buf, static_cast<std::size_t>(n));
    std::string out(static_cast<std::size_t>(n), '\0');
    va_start(args, format);
    std::vsnprintf(out.data(), out.size() + 1, format, args);
    va_end(args);
    return out;
}

std::string join(const std::vector<std::string>& parts, const char* sep) {
    std::string out;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        if (i) out += sep;
        out += parts[i];
    }
    return out;
}

}  // namespace

// ================================================================== base

std::optional<SafetyEvent> SafetyRule::evaluate(const SafetyContext&) const {
    return std::nullopt;
}
ControlCommand SafetyRule::enforce(ControlCommand cmd, const SafetyContext&) const {
    return cmd;
}
void SafetyRule::reset() {}

// ================================================================== TTC

std::optional<SafetyEvent> TTCRule::evaluate(const SafetyContext& ctx) const {
    if (!ctx.perception || ctx.perception->objects.empty()) return std::nullopt;
    const VehicleState& ego = ctx.ego;
    if (!all_finite({ego.x, ego.y, ego.yaw, ego.speed}))
        return std::nullopt;  // ego-state validity is the monitor's preflight job
    const double cy = std::cos(ego.yaw), sy = std::sin(ego.yaw);

    double best_ttc = kInf;
    std::string best_desc;
    for (const DetectedObject& obj : ctx.perception->objects) {
        const double px = obj.position.x, py = obj.position.y;
        if (!all_finite({px, py, obj.velocity.x, obj.velocity.y})) continue;
        const double dx = px - ego.x, dy = py - ego.y;
        const double fwd = dx * cy + dy * sy;
        const double lat = -dx * sy + dy * cy;
        if (fwd <= 0.0) continue;  // fully behind the bumper line
        const double corridor =
            kEgoHalfWidthM + std::max(0.5, std::abs(obj.bbox_extent.y));
        if (std::abs(lat) > corridor) continue;  // outside the swept path
        const double gap =
            fwd - kEgoHalfLengthM - std::max(0.0, std::abs(obj.bbox_extent.x));
        const double obj_speed_fwd = obj.velocity.x * cy + obj.velocity.y * sy;
        const double closing = ego.speed - obj_speed_fwd;
        double ttc;
        if (gap <= 0.0) {
            ttc = 0.0;  // already inside the safety envelope
        } else if (closing > MIN_CLOSING_MPS) {
            ttc = gap / closing;
        } else {
            continue;  // receding or pacing — no imminent collision
        }
        if (ttc < best_ttc) {
            best_ttc = ttc;
            best_desc = fmt("obj#%lld %s gap=%.1fm closing=%.1fm/s conf=%.2f",
                            static_cast<long long>(obj.obj_id), obj.cls.c_str(),
                            std::max(gap, 0.0), closing, obj.confidence);
        }
    }

    if (best_ttc == kInf) return std::nullopt;
    const double floor_s = ctx.cfg.min_ttc_s;
    if (best_ttc < floor_s)
        return SafetyEvent{"critical", name,
                           fmt("TTC %.2fs below floor %.2fs (%s)", best_ttc,
                               floor_s, best_desc.c_str())};
    if (best_ttc < floor_s * WARN_FACTOR)
        return SafetyEvent{"warning", name,
                           fmt("TTC %.2fs approaching floor %.2fs (%s)",
                               best_ttc, floor_s, best_desc.c_str())};
    return std::nullopt;
}

// ================================================================== speed

std::optional<SafetyEvent> SpeedLimitRule::evaluate(const SafetyContext& ctx) const {
    const double speed = ctx.ego.speed;
    const double limit = ctx.cfg.max_speed_mps;
    if (!finite(speed))
        return SafetyEvent{"critical", name, "ego speed is NaN/inf - telemetry invalid"};
    if (speed > limit * HARD_FACTOR)
        return SafetyEvent{"critical", name,
                           fmt("speed %.1fm/s exceeds hard limit %.1fm/s",
                               speed, limit * HARD_FACTOR)};
    if (speed > limit)
        return SafetyEvent{"warning", name,
                           fmt("speed %.1fm/s over limit %.1fm/s - clamping",
                               speed, limit)};
    return std::nullopt;
}

ControlCommand SpeedLimitRule::enforce(ControlCommand cmd, const SafetyContext& ctx) const {
    const SafetyConfig& cfg = ctx.cfg;
    const double speed = ctx.ego.speed;
    if (!finite(speed)) {
        cmd.throttle = 0.0;
        cmd.brake = 1.0;
        return cmd;
    }

    // Accel/decel budgets: bound the unitless demand by the nominal gains.
    const double thr_cap = clampf(cfg.max_accel_mps2 / kNominalMaxAccelMps2, 0.0, 1.0);
    const double brk_cap = clampf(cfg.max_brake_mps2 / kNominalMaxBrakeMps2, 0.0, 1.0);
    cmd.throttle = std::min(cmd.throttle, thr_cap);
    cmd.brake = std::min(cmd.brake, brk_cap);

    const double margin = cfg.max_speed_mps - speed;
    if (margin <= 0.0) {
        // Over the cap: kill throttle and add brake proportional to excess.
        cmd.throttle = 0.0;
        const double over_ratio = -margin / std::max(cfg.max_speed_mps, 1e-3);
        cmd.brake = std::max(cmd.brake, clampf(0.35 + 2.0 * over_ratio, 0.0, 1.0));
    } else if (margin < TAPER_BAND_MPS) {
        cmd.throttle *= clampf(margin / TAPER_BAND_MPS, 0.0, 1.0);
    }
    return cmd;
}

// ================================================================== space

std::optional<SafetyEvent> FreeSpaceRule::evaluate(const SafetyContext& ctx) const {
    if (!ctx.perception)
        return std::nullopt;  // absence of perception is flagged by the monitor itself
    const double free = ctx.perception->free_space_ahead;
    const double floor_m = ctx.cfg.min_free_space_m;
    if (!finite(free))
        return SafetyEvent{"critical", name,
                           "free-space reading is NaN/inf - treating as obstructed"};
    if (free < floor_m)
        return SafetyEvent{"critical", name,
                           fmt("free space %.1fm below minimum %.1fm", free, floor_m)};
    if (free < floor_m * WARN_FACTOR)
        return SafetyEvent{"warning", name,
                           fmt("free space %.1fm nearing minimum %.1fm", free, floor_m)};
    return std::nullopt;
}

// ================================================================== watchdog

std::optional<SafetyEvent> WatchdogRule::evaluate(const SafetyContext& ctx) const {
    const double timeout = ctx.cfg.watchdog_timeout_s;
    if (!ctx.pipeline_alive)
        return SafetyEvent{"critical", name,
                           "pipeline reports not alive - heartbeat lost"};
    const double hb_age = ctx.now - ctx.last_heartbeat;
    if (hb_age > timeout)
        return SafetyEvent{"critical", name,
                           fmt("pipeline heartbeat %.2fs old (timeout %.2fs)",
                               hb_age, timeout)};
    if (ctx.perception && finite(ctx.perception->timestamp)) {
        const double age = ctx.now - ctx.perception->timestamp;
        if (age > timeout)
            return SafetyEvent{"critical", name,
                               fmt("perception output %.2fs stale (timeout %.2fs)",
                                   age, timeout)};
    }
    if (finite(ctx.ego.timestamp)) {
        const double age = ctx.now - ctx.ego.timestamp;
        if (age > timeout)
            return SafetyEvent{"critical", name,
                               fmt("ego state %.2fs stale (timeout %.2fs)",
                                   age, timeout)};
    }
    if (ctx.dt > timeout * 2.0)
        return SafetyEvent{"warning", name,
                           fmt("safety cycle overran: %.0fms between checks",
                               ctx.dt * 1000.0)};
    return std::nullopt;
}

// ================================================================== steering

double SteerRateRule::baseline(const SafetyContext& ctx) const {
    if (ctx.applied && finite(ctx.applied->steer)) return ctx.applied->steer;
    if (finite(ctx.ego.steer)) return ctx.ego.steer;
    return 0.0;
}

std::optional<SafetyEvent> SteerRateRule::evaluate(const SafetyContext& ctx) const {
    if (ctx.dt <= 1e-4 || !finite(ctx.cmd.steer)) return std::nullopt;
    const double rate = std::abs(ctx.cmd.steer - baseline(ctx)) / ctx.dt;
    const double hard = ctx.cfg.max_steer_rate * 1.5;
    if (rate > hard)
        return SafetyEvent{"warning", name,
                           fmt("steer demand slews %.2f/s (limit %.2f/s) - rate-limiting",
                               rate, ctx.cfg.max_steer_rate)};
    return std::nullopt;
}

ControlCommand SteerRateRule::enforce(ControlCommand cmd, const SafetyContext& ctx) const {
    if (!finite(cmd.steer)) {
        cmd.steer = baseline(ctx);
        return cmd;
    }
    const double dt = std::max(ctx.dt, 1e-3);
    const double max_delta = ctx.cfg.max_steer_rate * dt;
    const double base = baseline(ctx);
    cmd.steer = clampf(cmd.steer, base - max_delta, base + max_delta);
    return cmd;
}

// ================================================================== lanes

std::optional<SafetyEvent> LaneDepartureRule::evaluate(const SafetyContext& ctx) const {
    if (!ctx.perception || !ctx.perception->lane) return std::nullopt;
    const LaneInfo& lane = *ctx.perception->lane;
    if (!lane.detected) {
        if (ctx.ego.speed > LOST_MIN_SPEED_MPS)
            return SafetyEvent{"warning", name,
                               fmt("lane tracking lost at %.1fm/s", ctx.ego.speed)};
        return std::nullopt;
    }
    const double half_w = lane.lane_width / 2.0;
    if (!all_finite({lane.center_offset, half_w}) || half_w <= 0.0)
        return std::nullopt;
    const double offset = std::abs(lane.center_offset);
    if (offset > half_w + EXIT_MARGIN_M)
        return SafetyEvent{"critical", name,
                           fmt("lane departure: |offset| %.2fm exceeds lane half-width %.2fm",
                               offset, half_w)};
    if (offset > 0.6 * half_w)
        return SafetyEvent{"warning", name,
                           fmt("drifting toward lane edge: |offset| %.2fm of %.2fm half-width",
                               offset, half_w)};
    return std::nullopt;
}

// ================================================================== defaults

std::vector<std::shared_ptr<SafetyRule>> default_rules() {
    return {
        std::make_shared<TTCRule>(),
        std::make_shared<SpeedLimitRule>(),
        std::make_shared<FreeSpaceRule>(),
        std::make_shared<WatchdogRule>(),
        std::make_shared<SteerRateRule>(),
        std::make_shared<LaneDepartureRule>(),
    };
}

// ================================================================== monitor

SafetyMonitor::SafetyMonitor(SafetyConfig cfg_,
                             std::vector<std::shared_ptr<SafetyRule>> rules_,
                             std::size_t event_history)
    : cfg(std::move(cfg_)),
      rules(rules_.empty() ? default_rules() : std::move(rules_)),
      event_history_(event_history) {
    const double now = clock_();
    mode_since_ = now;
    last_heartbeat_ = now;  // grace period == watchdog_timeout_s
}

// ---------------------------------------------------------------------- API

DriveMode SafetyMonitor::check(std::optional<VehicleState> ego,
                               std::optional<PerceptionOutput> perception,
                               std::optional<ControlCommand> cmd,
                               bool pipeline_alive) {
    try {
        return evaluate_cycle(std::move(ego), std::move(perception),
                              std::move(cmd), pipeline_alive);
    } catch (const std::exception& e) {
        log("error", std::string("internal monitor fault - fail-safe SAFE_STOP: ") + e.what());
    } catch (...) {
        log("error", "internal monitor fault - fail-safe SAFE_STOP");
    }
    mode_ = DriveMode::SAFE_STOP;
    if (latch_critical) latched_ = true;
    hazards_requested = true;
    return DriveMode::SAFE_STOP;
}

ControlCommand SafetyMonitor::enforce(std::optional<ControlCommand> cmd) {
    ControlCommand c = cmd ? *cmd : ControlCommand{};
    c = sanitize_cmd(c);
    if (mode_ == DriveMode::SAFE_STOP) return engage_safe_stop();
    if (mode_ == DriveMode::DISENGAGED)
        return c.clamp();  // human owns the car; still range-clamp the pass-through
    if (last_ctx_) {
        for (const auto& rule : rules) {
            if (!rule->enabled) continue;
            try {
                c = rule->enforce(c, *last_ctx_);
            } catch (const std::exception& e) {
                rule_faults_[rule->name] += 1;
                log_event(SafetyEvent{"critical", "rule." + rule->name,
                                      std::string("enforce() fault: ") + e.what() +
                                          " - skipping this shaper"});
            } catch (...) {
                rule_faults_[rule->name] += 1;
                log_event(SafetyEvent{"critical", "rule." + rule->name,
                                      "enforce() fault - skipping this shaper"});
            }
        }
    }
    return c.clamp();
}

ControlCommand SafetyMonitor::engage_safe_stop() {
    hazards_requested = true;
    double steer = 0.0;
    if (applied_ && finite(applied_->steer)) steer = applied_->steer;
    const double speed = last_ctx_ ? last_ctx_->ego.speed : kInf;
    const bool stopped = !finite(speed) || speed < 0.3;
    if (!safe_stop_announced_) {
        safe_stop_announced_ = true;
        log("warning", "SAFE STOP engaged - full brake, hazard lamps requested");
        log_event(SafetyEvent{"info", "monitor",
                              "safe stop engaged; hazard lamps requested"});
    }
    ControlCommand c;
    c.throttle = 0.0;
    c.brake = 1.0;
    c.steer = steer;
    c.hand_brake = stopped;  // hold the car once it has actually stopped
    c.reverse = false;
    return c;
}

void SafetyMonitor::disengage() {
    if (disengaged_) return;
    disengaged_ = true;
    const DriveMode prev = mode_;
    mode_ = DriveMode::DISENGAGED;
    mode_since_ = clock_();
    log("warning", std::string("drive mode ") + to_string(prev) +
                       " -> DISENGAGED (operator)");
    log_event(SafetyEvent{"info", "monitor", "disengaged by operator"});
}

void SafetyMonitor::reset() {
    const double now = clock_();
    mode_ = DriveMode::ENGAGED;
    mode_since_ = now;
    disengaged_ = false;
    latched_ = false;
    hazards_requested = false;
    safe_stop_announced_ = false;
    applied_ = std::nullopt;
    last_ctx_ = std::nullopt;
    last_check_ = std::nullopt;
    last_warn_t_ = -kInf;
    last_critical_t_ = -kInf;
    last_heartbeat_ = now;
    for (const auto& rule : rules) {
        try {
            rule->reset();
        } catch (...) {
            log("error", "rule " + rule->name + " failed to reset");
        }
    }
    log("info", "monitor reset -> ENGAGED");
    log_event(SafetyEvent{"info", "monitor", "monitor reset to ENGAGED"});
}

// ------------------------------------------------------------------ rule admin

bool SafetyMonitor::remove_rule(const std::string& name) {
    for (auto it = rules.begin(); it != rules.end(); ++it) {
        if ((*it)->name == name) {
            rules.erase(it);
            return true;
        }
    }
    return false;
}

bool SafetyMonitor::set_rule_enabled(const std::string& name, bool enabled) {
    for (const auto& rule : rules) {
        if (rule->name == name) {
            rule->enabled = enabled;
            return true;
        }
    }
    return false;
}

std::shared_ptr<SafetyRule> SafetyMonitor::get_rule(const std::string& name) const {
    for (const auto& rule : rules)
        if (rule->name == name) return rule;
    return nullptr;
}

// ------------------------------------------------------------------ introspect

std::vector<SafetyEvent> SafetyMonitor::recent_events(std::size_t n) const {
    const std::size_t count = events_.size();
    const std::size_t start = count > n ? count - n : 0;
    return {events_.begin() + static_cast<std::ptrdiff_t>(start), events_.end()};
}

MonitorStatus SafetyMonitor::status() const {
    const double now = clock_();
    MonitorStatus s;
    s.mode = to_string(mode_);
    s.mode_for_s = now - mode_since_;
    s.latched = latched_;
    s.hazards = hazards_requested;
    s.heartbeat_age_s = now - last_heartbeat_;
    s.rules.reserve(rules.size());
    for (const auto& r : rules) s.rules.emplace_back(r->name, r->enabled);
    s.rule_faults = rule_faults_;
    s.event_counts = counts_;
    s.events_logged = events_.size();
    return s;
}

// ------------------------------------------------------------------ internals

DriveMode SafetyMonitor::evaluate_cycle(std::optional<VehicleState> ego,
                                        std::optional<PerceptionOutput> perception,
                                        std::optional<ControlCommand> cmd,
                                        bool pipeline_alive) {
    const double now = clock_();
    const double dt =
        last_check_ ? std::min(std::max(now - *last_check_, 1e-4), 5.0)
                    : default_dt;
    last_check_ = now;

    std::vector<SafetyEvent> events;
    VehicleState ego_v = preflight_ego(std::move(ego), now, events);
    PerceptionOutput perception_v;
    if (!perception) {
        events.emplace_back("critical", "monitor.preflight",
                            "perception unavailable - treating world as obstructed");
        perception_v = blind_perception(now);
    } else {
        perception_v = std::move(*perception);
    }
    // In C++ the type system guarantees cmd is a ControlCommand; a missing
    // one (nullopt) means "no demand", same as Python's None branch.
    ControlCommand cmd_v = cmd ? *cmd : ControlCommand{};

    SafetyContext ctx;
    ctx.ego = ego_v;
    ctx.perception = std::move(perception_v);
    ctx.cmd = cmd_v;
    ctx.applied = applied_;
    ctx.pipeline_alive = pipeline_alive;
    ctx.now = now;
    ctx.dt = dt;
    ctx.cfg = cfg;
    ctx.last_heartbeat = last_heartbeat_;

    for (const auto& rule : rules) {
        if (!rule->enabled) continue;
        std::optional<SafetyEvent> ev;
        try {
            ev = rule->evaluate(ctx);
        } catch (const std::exception& e) {
            rule_faults_[rule->name] += 1;
            ev = SafetyEvent{"critical", "rule." + rule->name,
                             std::string("evaluate() fault: ") + e.what() +
                                 " - rule assumed compromised"};
        } catch (...) {
            rule_faults_[rule->name] += 1;
            ev = SafetyEvent{"critical", "rule." + rule->name,
                             "evaluate() fault - rule assumed compromised"};
        }
        if (ev) events.push_back(std::move(*ev));
    }

    for (const auto& ev : events) {
        log_event(ev);
        if (severity_order(ev.level) >= severity_order("critical"))
            last_critical_t_ = now;
        else if (ev.level == "warning")
            last_warn_t_ = now;
    }

    update_mode(arbitrate(events), now);
    last_ctx_ = ctx;
    applied_ = enforce(cmd_v);
    hazards_requested = hazards_requested || mode_ == DriveMode::SAFE_STOP;
    return mode_;
}

VehicleState SafetyMonitor::preflight_ego(std::optional<VehicleState> ego,
                                          double now,
                                          std::vector<SafetyEvent>& events) {
    if (!ego) {
        events.emplace_back("critical", "monitor.preflight",
                            "ego state missing - assuming stationary");
        VehicleState parked{};
        parked.timestamp = now;
        return parked;
    }
    VehicleState e = *ego;
    const std::pair<const char*, double*> fields[] = {
        {"x", &e.x},     {"y", &e.y},       {"yaw", &e.yaw},
        {"speed", &e.speed}, {"accel", &e.accel}, {"steer", &e.steer},
    };
    std::vector<std::string> bad;
    for (const auto& [fname, ptr] : fields)
        if (!finite(*ptr)) bad.emplace_back(fname);
    if (!bad.empty()) {
        events.emplace_back(
            "critical", "monitor.preflight",
            "ego telemetry non-finite fields: " + join(bad, ", ") +
                " - substituting zeros");
        for (const auto& [fname, ptr] : fields)
            if (!finite(*ptr)) *ptr = 0.0;
    } else if (e.speed < -0.5) {
        events.emplace_back("warning", "monitor.preflight",
                            fmt("negative speed %.1fm/s - sensor glitch or reversing",
                                e.speed));
    }
    return e;
}

ControlCommand SafetyMonitor::sanitize_cmd(ControlCommand c) {
    bool dirty = false;
    if (!finite(c.throttle)) { c.throttle = 0.0; dirty = true; }
    if (!finite(c.brake)) { c.brake = 0.0; dirty = true; }
    if (!finite(c.steer)) { c.steer = 0.0; dirty = true; }
    if (dirty)
        log_event(SafetyEvent{"warning", "monitor.sanitize",
                              "non-finite command fields zeroed before enforcement"});
    // hand_brake/reverse are real bools in C++ — no sanitation needed.
    return c;
}

PerceptionOutput SafetyMonitor::blind_perception(double now) {
    PerceptionOutput p;
    p.objects.clear();
    LaneInfo lane{};
    lane.detected = false;
    p.lane = lane;
    p.light = LightState::UNKNOWN;
    p.free_space_ahead = 0.0;  // blind == obstructed
    p.timestamp = now;
    return p;
}

DriveMode SafetyMonitor::arbitrate(const std::vector<SafetyEvent>& events) {
    int worst = 0;
    for (const auto& e : events) worst = std::max(worst, severity_order(e.level));
    if (worst >= severity_order("critical")) return DriveMode::SAFE_STOP;
    if (worst >= severity_order("warning")) return DriveMode::DEGRADED;
    return DriveMode::ENGAGED;
}

void SafetyMonitor::update_mode(DriveMode raw, double now) {
    if (disengaged_) return;  // operator owns the car; still recording events
    const DriveMode prev = mode_;
    if (mode_priority(raw) > mode_priority(prev)) {
        mode_ = raw;
    } else if (mode_priority(raw) < mode_priority(prev) && !latched_) {
        if (prev == DriveMode::SAFE_STOP &&
            now - last_critical_t_ > critical_clear_s) {
            mode_ = (now - last_warn_t_ <= warn_clear_s) ? DriveMode::DEGRADED
                                                        : DriveMode::ENGAGED;
        } else if (prev == DriveMode::DEGRADED &&
                   now - last_warn_t_ > warn_clear_s) {
            mode_ = DriveMode::ENGAGED;
        }
    }
    if (mode_ == DriveMode::SAFE_STOP && latch_critical) latched_ = true;
    if (mode_ != prev) {
        mode_since_ = now;
        log("info", std::string("drive mode ") + to_string(prev) + " -> " +
                        to_string(mode_));
    }
}

void SafetyMonitor::log_event(const SafetyEvent& ev) {
    events_.push_back(ev);
    if (events_.size() > event_history_) events_.pop_front();
    counts_[ev.level] += 1;
    if (logger_) logger_(ev.level, "[" + ev.source + "] " + ev.message);
}

void SafetyMonitor::log(const std::string& level, const std::string& message) const {
    if (logger_) logger_(level, message);
}

}  // namespace fsd
