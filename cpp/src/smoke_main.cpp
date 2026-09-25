// fsd_smoke — synthetic end-to-end exercise of the C++ core.
//
// Scenario: the ego car accelerates from rest along a straight reference path
// toward a stationary lead vehicle 60 m ahead. Perception is synthetic:
//   * the lead vehicle is reported as a DetectedObject each tick;
//   * an OccupancyGrid integrates a simulated sensor ray per tick and answers
//     the free_space_ahead query that feeds PerceptionOutput.
// The VehicleController tracks a 13.9 m/s target; the SafetyMonitor arbitrates
// the drive mode and the enforced command is what actually moves the ego.
//
// Expected timeline (20 Hz, dt = 0.05 s):
//   ENGAGED   ego accelerates toward the lead
//   DEGRADED  TTC warning (~2.7 s floor factor) as the gap shrinks
//   SAFE_STOP TTC below the 1.5 s floor -> latch, full brake, hazard request
//   stopped   hand brake holds the car; reset() -> falls back to DEGRADED
//             because free space is still under the warning band.
//
// Deterministic: the monitor runs on an injected clock (t = tick * dt).
// Exits 0 when every self-check passes, 1 otherwise.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "fsd/control.hpp"
#include "fsd/occupancy.hpp"
#include "fsd/safety.hpp"
#include "fsd/types.hpp"

using namespace fsd;

namespace {

constexpr double kDt = 0.05;           // 20 Hz
constexpr int kTicks = 200;            // 10 s of simulated time
constexpr double kLeadX = 60.0;        // stationary lead vehicle position
constexpr double kTargetSpeed = 13.9;  // m/s, ~50 km/h

// Nominal actuator model shared with the safety rules: throttle fraction ->
// ~6 m/s^2, brake fraction -> ~8 m/s^2, hand brake clamps velocity hard.
struct SimCar {
    double x = 0.0, y = 0.0, yaw = 0.0, v = 0.0, a = 0.0;
    static constexpr double kAccelPerThrottle = 6.0;
    static constexpr double kDecelPerBrake = 8.0;
    static constexpr double kMaxSteerRad = 60.0 * kPi / 180.0;
    static constexpr double kWheelbase = 2.875;

    void step(const ControlCommand& cmd, double dt) {
        const double delta = cmd.steer * kMaxSteerRad;
        a = cmd.throttle * kAccelPerThrottle - cmd.brake * kDecelPerBrake;
        if (cmd.hand_brake && v > 0.0) a = -12.0;
        v = std::max(0.0, v + a * dt);
        x += v * std::cos(yaw) * dt;
        y += v * std::sin(yaw) * dt;
        yaw += v / kWheelbase * std::tan(delta) * dt;
        if (v <= 0.0 && a < 0.0) a = 0.0;
    }
};

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::printf("  FAIL: %s\n", what);
    }
}

}  // namespace

int main() {
    double t = 0.0;  // synthetic clock domain

    SafetyConfig cfg;  // defaults: ttc 1.5 s, speed cap 16.7, free space 8 m
    SafetyMonitor monitor(cfg);
    monitor.set_clock([&t] { return t; });
    monitor.set_logger([](const std::string& level, const std::string& msg) {
        std::printf("      [%s] %s\n", level.c_str(), msg.c_str());
    });

    VehicleController controller(/*wheelbase_m=*/2.875, /*max_steer_deg=*/60.0,
                                 /*max_steer_rate=*/0.4,
                                 /*max_accel_mps2=*/3.0, /*max_brake_mps2=*/6.0,
                                 /*lookahead_m=*/8.0, /*use_mpc=*/true);

    // World window: x in [-10, 110], y in [-60, 60] at 0.25 m cells.
    OccupancyGrid grid(120.0, 120.0, 0.25, {-10.0, -60.0});

    // Straight reference path along +x.
    Trajectory traj;
    traj.target_speed = kTargetSpeed;
    for (double wx = -4.0; wx <= 120.0; wx += 2.0) {
        Waypoint p;
        p.x = wx;
        p.y = 0.0;
        p.yaw = 0.0;
        traj.points.push_back(p);
    }

    SimCar car;
    DriveMode prev_mode = monitor.mode();
    bool saw_degraded = false, saw_safe_stop = false;
    bool all_finite = true;
    double min_gap = kInf;

    std::printf("fsd_smoke: ego accelerates toward a stationary lead at x=%.0fm\n",
                kLeadX);
    std::printf("tick | mode       |    v  |  dist | free  |  thr |  brk | steer\n");
    std::printf("-----+------------+-------+-------+-------+------+------+------\n");

    for (int tick = 0; tick < kTicks; ++tick) {
        t = tick * kDt;

        // ---- synthetic perception ----------------------------------- //
        grid.clear();
        // Sensor ray from the ego to the lead vehicle's cell.
        grid.raycast(car.x, car.y, kLeadX, 0.0, /*hit=*/true);
        const double free_space =
            grid.free_space_ahead(car.x, car.y, car.yaw, 80.0, 1.0);

        PerceptionOutput perc;
        perc.timestamp = t;
        perc.free_space_ahead = free_space;
        perc.lane = LaneInfo{};  // detected, centered
        perc.lane->left_offset = 1.75;
        perc.lane->right_offset = -1.75;
        DetectedObject lead;
        lead.obj_id = 1;
        lead.cls = "vehicle";
        lead.position = Vec3{kLeadX, 0.0, 0.0};
        lead.velocity = Vec3{0.0, 0.0, 0.0};
        lead.bbox_extent = Vec3{2.2, 0.9, 0.8};
        lead.confidence = 0.95;
        lead.timestamp = t;
        perc.objects = {lead};

        VehicleState ego;
        ego.x = car.x;
        ego.y = car.y;
        ego.yaw = car.yaw;
        ego.speed = car.v;
        ego.accel = car.a;
        ego.steer = monitor.last_command() ? monitor.last_command()->steer : 0.0;
        ego.timestamp = t;

        // ---- control demand -> safety gate --------------------------- //
        const ControlCommand demand = controller.compute(&traj, ego, &perc);
        monitor.heartbeat();  // upstream pipeline still alive
        const DriveMode mode = monitor.check(ego, perc, demand,
                                             /*pipeline_alive=*/true);
        // NB: check() already ran enforce() — last_command() is what legally
        // leaves the gate this tick (it IS engage_safe_stop() in SAFE_STOP).
        const ControlCommand applied =
            monitor.last_command() ? *monitor.last_command()
                                   : monitor.engage_safe_stop();

        all_finite = all_finite && finite(applied.throttle) &&
                     finite(applied.brake) && finite(applied.steer);
        const double dist = kLeadX - car.x;
        min_gap = std::min(min_gap, dist);
        if (mode == DriveMode::DEGRADED) saw_degraded = true;
        if (mode == DriveMode::SAFE_STOP) saw_safe_stop = true;

        std::printf("%4d | %-10s | %5.2f | %5.1f | %5.1f | %4.2f | %4.2f | %+5.2f",
                    tick, to_string(mode), car.v, dist, free_space,
                    applied.throttle, applied.brake, applied.steer);
        if (mode != prev_mode)
            std::printf("  <== %s -> %s", to_string(prev_mode), to_string(mode));
        if (applied.hand_brake) std::printf("  [hand_brake]");
        std::printf("\n");
        prev_mode = mode;

        car.step(applied, kDt);
    }

    // ------------------------------------------------------------ checks
    std::printf("\n--- self-checks ---\n");
    expect(saw_degraded, "expected a DEGRADED phase (TTC warning band)");
    expect(saw_safe_stop, "expected SAFE_STOP to engage (TTC floor breach)");
    expect(monitor.latched(), "SAFE_STOP must latch until reset()");
    expect(monitor.mode() == DriveMode::SAFE_STOP, "mode should stay SAFE_STOP");
    expect(monitor.hazards_requested, "hazard lamps should be requested");
    expect(car.v < 0.3, "ego should be stopped after the safe stop");
    expect(min_gap > 4.6, "ego must never reach the lead vehicle's bbox");
    expect(all_finite, "all enforced commands must be finite");
    const auto counts = monitor.event_counts();
    const auto crit_it = counts.find("critical");
    expect(crit_it != counts.end() && crit_it->second > 0,
           "expected at least one critical event logged");
    std::printf("stopped at %.1f m behind the lead (min gap %.1f m)\n",
                kLeadX - car.x, min_gap);

    // reset() must clear the latch — but the lead is still parked ahead, so
    // the very next check drops back to DEGRADED (free-space warning band).
    t += kDt;  // advance the synthetic clock for the extra check
    monitor.reset();
    expect(monitor.mode() == DriveMode::ENGAGED, "reset() should clear to ENGAGED");
    VehicleState ego;
    ego.x = car.x; ego.y = car.y; ego.yaw = car.yaw; ego.speed = car.v;
    ego.timestamp = t;
    PerceptionOutput perc;
    perc.timestamp = t;
    perc.free_space_ahead = grid.free_space_ahead(car.x, car.y, car.yaw, 80.0, 1.0);
    perc.lane = LaneInfo{};
    DetectedObject lead;
    lead.obj_id = 1; lead.cls = "vehicle";
    lead.position = Vec3{kLeadX, 0.0, 0.0};
    lead.bbox_extent = Vec3{2.2, 0.9, 0.8};
    lead.confidence = 0.95;
    lead.timestamp = t;
    perc.objects = {lead};
    monitor.heartbeat();
    const DriveMode post_reset =
        monitor.check(ego, perc, ControlCommand{}, true);
    std::printf("post-reset check: %s (lead still %.1f m ahead)\n",
                to_string(post_reset), kLeadX - car.x);
    expect(post_reset == DriveMode::DEGRADED ||
               post_reset == DriveMode::SAFE_STOP,
           "post-reset check should still refuse ENGAGED near the lead");

    const MonitorStatus st = monitor.status();
    std::printf("\nstatus: mode=%s latched=%d hazards=%d events=%zu\n",
                st.mode.c_str(), static_cast<int>(st.latched),
                static_cast<int>(st.hazards), st.events_logged);
    for (const auto& [lvl, n] : st.event_counts)
        std::printf("  events[%s] = %d\n", lvl.c_str(), n);

    if (failures == 0) {
        std::printf("\nfsd_smoke: PASS\n");
        return 0;
    }
    std::printf("\nfsd_smoke: %d check(s) FAILED\n", failures);
    return 1;
}
