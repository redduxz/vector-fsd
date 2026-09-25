// pybind11 bindings for the fsd C++ core — module name ``fsd_cpp``.
//
// Guarded by FSD_WITH_PYBIND so the file is a legal (empty) translation unit
// when compiled without pybind11 — e.g. inside the fsd_core static library.
// The CMake option FSD_PYBIND defines it for the fsd_cpp extension target.
#ifdef FSD_WITH_PYBIND

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "fsd/control.hpp"
#include "fsd/occupancy.hpp"
#include "fsd/safety.hpp"
#include "fsd/types.hpp"

namespace py = pybind11;
using namespace fsd;

// The waypoint/object payload vectors are bound as opaque, list-like types so
// struct members (Trajectory::points, PerceptionOutput::objects) can expose
// *live* views — appending on the Python side mutates the C++ object. Without
// MAKE_OPAQUE the stl.h caster would silently hand out list copies.
PYBIND11_MAKE_OPAQUE(std::vector<fsd::Waypoint>);
PYBIND11_MAKE_OPAQUE(std::vector<fsd::DetectedObject>);

namespace {

/// Trampoline so Python subclasses of SafetyRule can override the virtuals.
/// NB: Python-side rule objects must stay alive while the monitor uses them —
/// the add_rule/ctor bindings use keep_alive for exactly this.
class PySafetyRule : public SafetyRule {
public:
    using SafetyRule::SafetyRule;
    std::optional<SafetyEvent> evaluate(const SafetyContext& ctx) const override {
        PYBIND11_OVERRIDE(std::optional<SafetyEvent>, SafetyRule, evaluate, ctx);
    }
    ControlCommand enforce(ControlCommand cmd, const SafetyContext& ctx) const override {
        PYBIND11_OVERRIDE(ControlCommand, SafetyRule, enforce, cmd, ctx);
    }
    void reset() override { PYBIND11_OVERRIDE(void, SafetyRule, reset, ); }
};

VehicleState make_vehicle_state(double x, double y, double z, double yaw,
                                double speed, double accel, double steer,
                                std::optional<double> timestamp) {
    VehicleState v;
    v.x = x;
    v.y = y;
    v.z = z;
    v.yaw = yaw;
    v.speed = speed;
    v.accel = accel;
    v.steer = steer;
    v.timestamp = timestamp.value_or(now_seconds());
    return v;
}

DetectedObject make_detected_object(int64_t obj_id, std::string cls,
                                    Vec3 position, Vec3 velocity,
                                    Vec3 bbox_extent, double confidence,
                                    std::optional<double> timestamp) {
    DetectedObject o;
    o.obj_id = obj_id;
    o.cls = std::move(cls);
    o.position = position;
    o.velocity = velocity;
    o.bbox_extent = bbox_extent;
    o.confidence = confidence;
    o.timestamp = timestamp.value_or(now_seconds());
    return o;
}

/// Accept any iterable of T (list, tuple, or the bound vector type itself).
template <typename T>
std::vector<T> iter_to_vector(py::iterable it) {
    std::vector<T> out;
    for (py::handle h : it) out.push_back(h.cast<T>());
    return out;
}

PerceptionOutput make_perception(py::iterable objects,
                                 std::optional<LaneInfo> lane, LightState light,
                                 double free_space_ahead,
                                 std::optional<double> timestamp) {
    PerceptionOutput p;
    p.objects = iter_to_vector<DetectedObject>(objects);
    p.lane = std::move(lane);
    p.light = light;
    p.free_space_ahead = free_space_ahead;
    p.timestamp = timestamp.value_or(now_seconds());
    return p;
}

SafetyEvent make_event(std::string level, std::string source,
                       std::string message, std::optional<double> timestamp) {
    SafetyEvent e(std::move(level), std::move(source), std::move(message));
    e.timestamp = timestamp.value_or(now_seconds());
    return e;
}

}  // namespace

PYBIND11_MODULE(fsd_cpp, m) {
    m.doc() = "fsd_cpp — C++ hot-path core for the FSD stack (Phase 1 port).";

    // --------------------------------------------------------------- enums
    py::enum_<LaneChangeState>(m, "LaneChangeState")
        .value("KEEP", LaneChangeState::KEEP)
        .value("LEFT", LaneChangeState::LEFT)
        .value("RIGHT", LaneChangeState::RIGHT);

    py::enum_<LightState>(m, "LightState")
        .value("RED", LightState::RED)
        .value("YELLOW", LightState::YELLOW)
        .value("GREEN", LightState::GREEN)
        .value("UNKNOWN", LightState::UNKNOWN);

    py::enum_<DriveMode>(m, "DriveMode")
        .value("ENGAGED", DriveMode::ENGAGED)
        .value("DEGRADED", DriveMode::DEGRADED)
        .value("SAFE_STOP", DriveMode::SAFE_STOP)
        .value("DISENGAGED", DriveMode::DISENGAGED);

    m.attr("SEVERITY_ORDER") =
        py::dict(py::arg("info") = 0, py::arg("warning") = 1,
                 py::arg("critical") = 2);
    m.def("now_seconds", &now_seconds, "Wall-clock seconds (time.time domain).");

    // --------------------------------------------------------------- types
    py::class_<Vec3>(m, "Vec3")
        .def(py::init<double, double, double>(), py::arg("x"),
             py::arg("y"), py::arg("z") = 0.0)
        .def_readwrite("x", &Vec3::x)
        .def_readwrite("y", &Vec3::y)
        .def_readwrite("z", &Vec3::z)
        .def("norm", &Vec3::norm);

    py::class_<VehicleState>(m, "VehicleState")
        .def(py::init(&make_vehicle_state), py::arg("x"), py::arg("y"),
             py::arg("z"), py::arg("yaw"), py::arg("speed"), py::arg("accel"),
             py::arg("steer"), py::arg("timestamp") = py::none())
        .def_readwrite("x", &VehicleState::x)
        .def_readwrite("y", &VehicleState::y)
        .def_readwrite("z", &VehicleState::z)
        .def_readwrite("yaw", &VehicleState::yaw)
        .def_readwrite("speed", &VehicleState::speed)
        .def_readwrite("accel", &VehicleState::accel)
        .def_readwrite("steer", &VehicleState::steer)
        .def_readwrite("timestamp", &VehicleState::timestamp);

    py::class_<Waypoint>(m, "Waypoint")
        .def(py::init<double, double, double, double, double>(),
             py::arg("x"), py::arg("y"), py::arg("z") = 0.0,
             py::arg("yaw") = 0.0, py::arg("speed_limit") = 13.9)
        .def_readwrite("x", &Waypoint::x)
        .def_readwrite("y", &Waypoint::y)
        .def_readwrite("z", &Waypoint::z)
        .def_readwrite("yaw", &Waypoint::yaw)
        .def_readwrite("speed_limit", &Waypoint::speed_limit);

    py::class_<Trajectory>(m, "Trajectory")
        .def(py::init([](py::iterable points, double target_speed,
                         double horizon_s) {
                 Trajectory t;
                 t.points = iter_to_vector<Waypoint>(points);
                 t.target_speed = target_speed;
                 t.horizon_s = horizon_s;
                 return t;
             }),
             py::arg("points"), py::arg("target_speed"),
             py::arg("horizon_s") = 4.0)
        .def_property(
            "points",
            // Live view of the member vector (opaque bound type).
            [](py::object self) {
                return py::cast(&self.cast<Trajectory&>().points,
                                py::return_value_policy::reference, self);
            },
            [](Trajectory& t, py::iterable v) {
                t.points = iter_to_vector<Waypoint>(v);
            })
        .def_readwrite("target_speed", &Trajectory::target_speed)
        .def_readwrite("horizon_s", &Trajectory::horizon_s)
        .def("empty", &Trajectory::empty);

    py::class_<DetectedObject>(m, "DetectedObject")
        .def(py::init(&make_detected_object), py::arg("obj_id"), py::arg("cls"),
             py::arg("position"), py::arg("velocity"), py::arg("bbox_extent"),
             py::arg("confidence"), py::arg("timestamp") = py::none())
        .def_readwrite("obj_id", &DetectedObject::obj_id)
        .def_readwrite("cls", &DetectedObject::cls)
        .def_readwrite("position", &DetectedObject::position)
        .def_readwrite("velocity", &DetectedObject::velocity)
        .def_readwrite("bbox_extent", &DetectedObject::bbox_extent)
        .def_readwrite("confidence", &DetectedObject::confidence)
        .def_readwrite("timestamp", &DetectedObject::timestamp);

    // Bound vector types (opaque) so payload fields expose *live* list-like
    // views — traj.points.append(wp) mutates the trajectory itself.
    py::bind_vector<std::vector<Waypoint>>(m, "WaypointList")
        .def(py::init([](py::iterable it) { return iter_to_vector<Waypoint>(it); }),
             py::arg("items"));
    py::bind_vector<std::vector<DetectedObject>>(m, "DetectedObjectList")
        .def(py::init([](py::iterable it) {
            return iter_to_vector<DetectedObject>(it);
        }), py::arg("items"));

    py::class_<LaneInfo>(m, "LaneInfo")
        .def(py::init<double, double, double, double, double, double, bool>(),
             py::arg("left_offset"), py::arg("right_offset"),
             py::arg("center_offset"), py::arg("heading_error"),
             py::arg("curvature"), py::arg("lane_width") = 3.5,
             py::arg("detected") = true)
        .def_readwrite("left_offset", &LaneInfo::left_offset)
        .def_readwrite("right_offset", &LaneInfo::right_offset)
        .def_readwrite("center_offset", &LaneInfo::center_offset)
        .def_readwrite("heading_error", &LaneInfo::heading_error)
        .def_readwrite("curvature", &LaneInfo::curvature)
        .def_readwrite("lane_width", &LaneInfo::lane_width)
        .def_readwrite("detected", &LaneInfo::detected);

    py::class_<ControlCommand>(m, "ControlCommand")
        .def(py::init<double, double, double, bool, bool>(),
             py::arg("throttle") = 0.0, py::arg("brake") = 0.0,
             py::arg("steer") = 0.0, py::arg("hand_brake") = false,
             py::arg("reverse") = false)
        .def_readwrite("throttle", &ControlCommand::throttle)
        .def_readwrite("brake", &ControlCommand::brake)
        .def_readwrite("steer", &ControlCommand::steer)
        .def_readwrite("hand_brake", &ControlCommand::hand_brake)
        .def_readwrite("reverse", &ControlCommand::reverse)
        .def("clamp", &ControlCommand::clamp);

    py::class_<SafetyEvent>(m, "SafetyEvent")
        .def(py::init(&make_event), py::arg("level"), py::arg("source"),
             py::arg("message"), py::arg("timestamp") = py::none())
        .def_readwrite("level", &SafetyEvent::level)
        .def_readwrite("source", &SafetyEvent::source)
        .def_readwrite("message", &SafetyEvent::message)
        .def_readwrite("timestamp", &SafetyEvent::timestamp);

    py::class_<PerceptionOutput>(m, "PerceptionOutput")
        .def(py::init(&make_perception), py::arg("objects"), py::arg("lane"),
             py::arg("light") = LightState::UNKNOWN,
             py::arg("free_space_ahead") = 100.0,
             py::arg("timestamp") = py::none())
        .def_property(
            "objects",
            [](py::object self) {
                return py::cast(&self.cast<PerceptionOutput&>().objects,
                                py::return_value_policy::reference, self);
            },
            [](PerceptionOutput& p, py::iterable v) {
                p.objects = iter_to_vector<DetectedObject>(v);
            })
        .def_readwrite("lane", &PerceptionOutput::lane)
        .def_readwrite("light", &PerceptionOutput::light)
        .def_readwrite("free_space_ahead", &PerceptionOutput::free_space_ahead)
        .def_readwrite("timestamp", &PerceptionOutput::timestamp);

    py::class_<SafetyConfig>(m, "SafetyConfig")
        .def(py::init<>())
        .def_readwrite("min_ttc_s", &SafetyConfig::min_ttc_s)
        .def_readwrite("max_speed_mps", &SafetyConfig::max_speed_mps)
        .def_readwrite("max_accel_mps2", &SafetyConfig::max_accel_mps2)
        .def_readwrite("max_brake_mps2", &SafetyConfig::max_brake_mps2)
        .def_readwrite("watchdog_timeout_s", &SafetyConfig::watchdog_timeout_s)
        .def_readwrite("min_free_space_m", &SafetyConfig::min_free_space_m)
        .def_readwrite("max_steer_rate", &SafetyConfig::max_steer_rate);

    // --------------------------------------------------------------- safety
    py::class_<SafetyContext>(m, "SafetyContext")
        .def(py::init([](VehicleState ego, std::optional<PerceptionOutput> perception,
                         ControlCommand cmd, std::optional<ControlCommand> applied,
                         bool pipeline_alive, std::optional<double> now, double dt,
                         SafetyConfig cfg, std::optional<double> last_heartbeat) {
                 SafetyContext c;
                 c.ego = ego;
                 c.perception = std::move(perception);
                 c.cmd = cmd;
                 c.applied = applied;
                 c.pipeline_alive = pipeline_alive;
                 c.now = now.value_or(now_seconds());
                 c.dt = dt;
                 c.cfg = cfg;
                 c.last_heartbeat = last_heartbeat.value_or(now_seconds());
                 return c;
             }),
             py::arg("ego"), py::arg("perception") = py::none(),
             py::arg("cmd") = ControlCommand{}, py::arg("applied") = py::none(),
             py::arg("pipeline_alive") = true, py::arg("now") = py::none(),
             py::arg("dt") = 0.05, py::arg("cfg") = SafetyConfig{},
             py::arg("last_heartbeat") = py::none())
        .def_readwrite("ego", &SafetyContext::ego)
        .def_readwrite("perception", &SafetyContext::perception)
        .def_readwrite("cmd", &SafetyContext::cmd)
        .def_readwrite("applied", &SafetyContext::applied)
        .def_readwrite("pipeline_alive", &SafetyContext::pipeline_alive)
        .def_readwrite("now", &SafetyContext::now)
        .def_readwrite("dt", &SafetyContext::dt)
        .def_readwrite("cfg", &SafetyContext::cfg)
        .def_readwrite("last_heartbeat", &SafetyContext::last_heartbeat);

    m.attr("NOMINAL_MAX_ACCEL_MPS2") = kNominalMaxAccelMps2;
    m.attr("NOMINAL_MAX_BRAKE_MPS2") = kNominalMaxBrakeMps2;

    py::class_<SafetyRule, PySafetyRule, std::shared_ptr<SafetyRule>>(m,
                                                                    "SafetyRule")
        .def(py::init<>())
        .def_readwrite("name", &SafetyRule::name)
        .def_readwrite("enabled", &SafetyRule::enabled)
        .def("evaluate", &SafetyRule::evaluate, py::arg("ctx"))
        .def("enforce", &SafetyRule::enforce, py::arg("cmd"), py::arg("ctx"))
        .def("reset", &SafetyRule::reset);

    py::class_<TTCRule, SafetyRule, std::shared_ptr<TTCRule>>(m, "TTCRule")
        .def(py::init<>())
        .def_readonly_static("MIN_CLOSING_MPS", &TTCRule::MIN_CLOSING_MPS)
        .def_readonly_static("WARN_FACTOR", &TTCRule::WARN_FACTOR);
    py::class_<SpeedLimitRule, SafetyRule, std::shared_ptr<SpeedLimitRule>>(
        m, "SpeedLimitRule")
        .def(py::init<>())
        .def_readonly_static("TAPER_BAND_MPS", &SpeedLimitRule::TAPER_BAND_MPS)
        .def_readonly_static("HARD_FACTOR", &SpeedLimitRule::HARD_FACTOR);
    py::class_<FreeSpaceRule, SafetyRule, std::shared_ptr<FreeSpaceRule>>(
        m, "FreeSpaceRule")
        .def(py::init<>())
        .def_readonly_static("WARN_FACTOR", &FreeSpaceRule::WARN_FACTOR);
    py::class_<WatchdogRule, SafetyRule, std::shared_ptr<WatchdogRule>>(
        m, "WatchdogRule")
        .def(py::init<>());
    py::class_<SteerRateRule, SafetyRule, std::shared_ptr<SteerRateRule>>(
        m, "SteerRateRule")
        .def(py::init<>());
    py::class_<LaneDepartureRule, SafetyRule,
               std::shared_ptr<LaneDepartureRule>>(m, "LaneDepartureRule")
        .def(py::init<>())
        .def_readonly_static("LOST_MIN_SPEED_MPS",
                             &LaneDepartureRule::LOST_MIN_SPEED_MPS)
        .def_readonly_static("EXIT_MARGIN_M", &LaneDepartureRule::EXIT_MARGIN_M);

    m.def("default_rules", &default_rules,
          "The standard rule set, in enforcement order.");

    py::class_<MonitorStatus>(m, "MonitorStatus")
        .def_readonly("mode", &MonitorStatus::mode)
        .def_readonly("mode_for_s", &MonitorStatus::mode_for_s)
        .def_readonly("latched", &MonitorStatus::latched)
        .def_readonly("hazards", &MonitorStatus::hazards)
        .def_readonly("heartbeat_age_s", &MonitorStatus::heartbeat_age_s)
        .def_readonly("rules", &MonitorStatus::rules)
        .def_readonly("rule_faults", &MonitorStatus::rule_faults)
        .def_readonly("event_counts", &MonitorStatus::event_counts)
        .def_readonly("events_logged", &MonitorStatus::events_logged);

    py::class_<SafetyMonitor>(m, "SafetyMonitor")
        .def(py::init<SafetyConfig, std::vector<std::shared_ptr<SafetyRule>>,
                      std::size_t>(),
             py::arg("cfg") = SafetyConfig{},
             py::arg("rules") = std::vector<std::shared_ptr<SafetyRule>>{},
             py::arg("event_history") = 4096,
             // keep python-defined rules alive while the monitor lives
             py::keep_alive<1, 3>())
        .def_readwrite("cfg", &SafetyMonitor::cfg)
        .def_readwrite("rules", &SafetyMonitor::rules)
        .def_readwrite("latch_critical", &SafetyMonitor::latch_critical)
        .def_readwrite("warn_clear_s", &SafetyMonitor::warn_clear_s)
        .def_readwrite("critical_clear_s", &SafetyMonitor::critical_clear_s)
        .def_readwrite("default_dt", &SafetyMonitor::default_dt)
        .def_readwrite("hazards_requested", &SafetyMonitor::hazards_requested)
        .def("check", &SafetyMonitor::check, py::arg("ego") = py::none(),
             py::arg("perception") = py::none(), py::arg("cmd") = py::none(),
             py::arg("pipeline_alive") = true)
        .def("enforce", &SafetyMonitor::enforce, py::arg("cmd") = py::none())
        .def("engage_safe_stop", &SafetyMonitor::engage_safe_stop)
        .def("heartbeat", &SafetyMonitor::heartbeat)
        .def("disengage", &SafetyMonitor::disengage)
        .def("reset", &SafetyMonitor::reset)
        .def("add_rule", &SafetyMonitor::add_rule, py::arg("rule"),
             // keep the (possibly python-defined) rule alive with the monitor
             py::keep_alive<1, 2>())
        .def("remove_rule", &SafetyMonitor::remove_rule, py::arg("name"))
        .def("set_rule_enabled", &SafetyMonitor::set_rule_enabled,
             py::arg("name"), py::arg("enabled"))
        .def("get_rule", &SafetyMonitor::get_rule, py::arg("name"))
        .def("recent_events", &SafetyMonitor::recent_events,
             py::arg("n") = 20)
        .def("status", &SafetyMonitor::status)
        .def("set_clock", [](SafetyMonitor& self, py::object fn) {
            self.set_clock([fn]() -> double {
                py::gil_scoped_acquire gil;
                return fn().cast<double>();
            });
        }, py::arg("fn"), "Inject a synthetic clock: callable() -> seconds.")
        .def("set_logger", [](SafetyMonitor& self, py::object fn) {
            self.set_logger([fn](const std::string& level,
                                 const std::string& message) {
                py::gil_scoped_acquire gil;
                fn(level, message);
            });
        }, py::arg("fn"), "Install a log sink: callable(level, message).")
        .def_property_readonly("mode", &SafetyMonitor::mode)
        .def_property_readonly("latched", &SafetyMonitor::latched)
        .def_property_readonly("last_command", &SafetyMonitor::last_command)
        .def_property_readonly("event_log", &SafetyMonitor::event_log)
        .def_property_readonly("event_counts", &SafetyMonitor::event_counts)
        .def_property_readonly("healthy", &SafetyMonitor::healthy);

    // -------------------------------------------------------------- control
    py::class_<PIDController>(m, "PIDController")
        .def(py::init<double, double, double, double, double, double, double,
                      double, std::string>(),
             py::arg("kp") = 1.0, py::arg("ki") = 0.0, py::arg("kd") = 0.0,
             py::arg("i_min") = -1.0, py::arg("i_max") = 1.0,
             py::arg("out_min") = -kInf, py::arg("out_max") = kInf,
             py::arg("deriv_tau") = 0.05, py::arg("name") = "pid")
        .def_readwrite("kp", &PIDController::kp)
        .def_readwrite("ki", &PIDController::ki)
        .def_readwrite("kd", &PIDController::kd)
        .def_readwrite("i_min", &PIDController::i_min)
        .def_readwrite("i_max", &PIDController::i_max)
        .def_readwrite("out_min", &PIDController::out_min)
        .def_readwrite("out_max", &PIDController::out_max)
        .def_readwrite("deriv_tau", &PIDController::deriv_tau)
        .def_readwrite("name", &PIDController::name)
        .def("reset", &PIDController::reset)
        .def("step", &PIDController::step, py::arg("error"), py::arg("dt"))
        .def("__call__", &PIDController::step, py::arg("error"), py::arg("dt"))
        .def_property_readonly("integral", &PIDController::integral);

    py::class_<PurePursuit>(m, "PurePursuit")
        .def(py::init<double, double>(), py::arg("wheelbase_m") = 2.875,
             py::arg("lookahead_gain") = 0.35)
        .def_readwrite("wheelbase", &PurePursuit::wheelbase)
        .def_readwrite("lookahead_gain", &PurePursuit::lookahead_gain)
        .def("steer_angle", &PurePursuit::steer_angle, py::arg("px"),
             py::arg("py"), py::arg("ego"), py::arg("lookahead_m"))
        .def("steer", &PurePursuit::steer, py::arg("traj"), py::arg("ego"),
             py::arg("lookahead_m"), py::arg("max_steer_rad"));

    py::class_<StanleyLateral>(m, "StanleyLateral")
        .def(py::init<double, double, double, double>(),
             py::arg("wheelbase_m") = 2.875, py::arg("max_steer_deg") = 60.0,
             py::arg("k_stanley") = 2.5, py::arg("k_soft_mps") = 1.2)
        .def_readwrite("wheelbase", &StanleyLateral::wheelbase)
        .def_readwrite("max_steer_rad", &StanleyLateral::max_steer_rad)
        .def_readwrite("k_stanley", &StanleyLateral::k_stanley)
        .def_readwrite("k_soft", &StanleyLateral::k_soft)
        .def("steer_rad", &StanleyLateral::steer_rad, py::arg("theta_path"),
             py::arg("ego_yaw"), py::arg("e_fa"), py::arg("speed"))
        .def("steer", &StanleyLateral::steer, py::arg("traj"), py::arg("ego"));

    py::class_<LateralController>(m, "LateralController")
        .def(py::init<double, double, double, double, double, double>(),
             py::arg("wheelbase_m") = 2.875, py::arg("max_steer_deg") = 60.0,
             py::arg("k_stanley") = 2.5, py::arg("k_soft_mps") = 1.2,
             py::arg("low_speed_mps") = 0.6, py::arg("lookahead_gain") = 0.35)
        .def_readwrite("wheelbase", &LateralController::wheelbase)
        .def_readwrite("max_steer_rad", &LateralController::max_steer_rad)
        .def_readwrite("k_stanley", &LateralController::k_stanley)
        .def_readwrite("k_soft", &LateralController::k_soft)
        .def_readwrite("low_speed", &LateralController::low_speed)
        .def_readwrite("lookahead_gain", &LateralController::lookahead_gain)
        .def("steer", &LateralController::steer, py::arg("traj"),
             py::arg("ego"), py::arg("lookahead_m") = 8.0);

    py::class_<LongitudinalController>(m, "LongitudinalController")
        .def(py::init<double, double, double, double, double, double, double,
                      double, double, double, double>(),
             py::arg("kp") = 1.2, py::arg("ki") = 0.25, py::arg("kd") = 0.05,
             py::arg("max_accel_mps2") = 3.0, py::arg("max_brake_mps2") = 6.0,
             py::arg("v_max_mps") = 55.0, py::arg("accel_deadband") = 0.08,
             py::arg("stop_margin_m") = 2.0, py::arg("min_ttc_s") = 1.2,
             py::arg("brake_hold") = 0.35, py::arg("ego_half_length_m") = 2.4)
        .def_readwrite("max_accel", &LongitudinalController::max_accel)
        .def_readwrite("max_brake", &LongitudinalController::max_brake)
        .def_readwrite("v_max", &LongitudinalController::v_max)
        .def_readwrite("deadband", &LongitudinalController::deadband)
        .def_readwrite("stop_margin", &LongitudinalController::stop_margin)
        .def_readwrite("min_ttc", &LongitudinalController::min_ttc)
        .def_readwrite("brake_hold", &LongitudinalController::brake_hold)
        .def_readwrite("ego_half_length",
                       &LongitudinalController::ego_half_length)
        .def_readwrite("pid", &LongitudinalController::pid)
        .def("accel_cmd", &LongitudinalController::accel_cmd,
             py::arg("target_speed"), py::arg("ego"),
             py::arg("perception") = py::none())
        .def("accel_to_command", &LongitudinalController::accel_to_command,
             py::arg("a_des"), py::arg("speed"));

    py::class_<MPCResult>(m, "MPCResult")
        .def_readonly("steer_norm", &MPCResult::steer_norm)
        .def_readonly("accel", &MPCResult::accel);

    py::class_<MPCLite>(m, "MPCLite")
        .def(py::init<double, double, double, double, double, double, double,
                      double, double, double, double, double, int>(),
             py::arg("wheelbase_m") = 2.875, py::arg("max_steer_deg") = 60.0,
             py::arg("horizon_s") = 1.2, py::arg("dt") = 0.1,
             py::arg("v_max_mps") = 16.7, py::arg("max_accel_mps2") = 3.0,
             py::arg("max_brake_mps2") = 6.0, py::arg("w_cte") = 6.0,
             py::arg("w_yaw") = 1.5, py::arg("w_vel") = 0.4,
             py::arg("w_effort") = 0.15, py::arg("w_terminal") = 2.0,
             py::arg("refine_passes") = 1)
        .def("optimize", &MPCLite::optimize, py::arg("traj"), py::arg("ego"),
             py::arg("target_speed"), py::arg("base_steer_norm") = 0.0,
             py::arg("base_accel") = 0.0);

    py::class_<VehicleController>(m, "VehicleController")
        .def(py::init<double, double, double, double, double, double, bool,
                      double, double>(),
             py::arg("wheelbase_m") = 2.875, py::arg("max_steer_deg") = 60.0,
             py::arg("max_steer_rate") = 0.4, py::arg("max_accel_mps2") = 3.0,
             py::arg("max_brake_mps2") = 6.0, py::arg("lookahead_m") = 8.0,
             py::arg("use_mpc") = true, py::arg("mpc_blend") = 0.35,
             py::arg("mpc_min_speed") = 1.0)
        .def_readwrite("lateral", &VehicleController::lateral)
        .def_readwrite("longitudinal", &VehicleController::longitudinal)
        .def_readwrite("mpc", &VehicleController::mpc)
        .def_readwrite("lookahead_m", &VehicleController::lookahead_m)
        .def_readwrite("max_steer_rate", &VehicleController::max_steer_rate)
        .def_readwrite("use_mpc", &VehicleController::use_mpc)
        .def_readwrite("mpc_blend", &VehicleController::mpc_blend)
        .def_readwrite("mpc_min_speed", &VehicleController::mpc_min_speed)
        .def_readwrite("max_accel", &VehicleController::max_accel)
        .def_readwrite("max_brake", &VehicleController::max_brake)
        .def("compute", &VehicleController::compute, py::arg("trajectory"),
             py::arg("ego"), py::arg("perception") = py::none())
        .def("step", &VehicleController::step, py::arg("trajectory"),
             py::arg("ego"), py::arg("perception") = py::none())
        .def("reset", &VehicleController::reset);

    // ------------------------------------------------------------ occupancy
    py::class_<OccupancyGrid>(m, "OccupancyGrid")
        .def(py::init<double, double, double, std::pair<double, double>>(),
             py::arg("width_m") = 120.0, py::arg("height_m") = 120.0,
             py::arg("resolution") = 0.25,
             py::arg("origin") = std::pair<double, double>{0.0, 0.0})
        .def_readwrite("res", &OccupancyGrid::res)
        .def_readwrite("ox", &OccupancyGrid::ox)
        .def_readwrite("oy", &OccupancyGrid::oy)
        .def_readwrite("lo_hit", &OccupancyGrid::lo_hit)
        .def_readwrite("lo_free", &OccupancyGrid::lo_free)
        .def_readwrite("lo_min", &OccupancyGrid::lo_min)
        .def_readwrite("lo_max", &OccupancyGrid::lo_max)
        .def_readonly("w", &OccupancyGrid::w)
        .def_readonly("h", &OccupancyGrid::h)
        .def("clear", &OccupancyGrid::clear)
        .def("set_origin", &OccupancyGrid::set_origin, py::arg("ox"),
             py::arg("oy"), py::arg("keep") = true)
        .def("cell", &OccupancyGrid::cell, py::arg("x"), py::arg("y"))
        .def("in_bounds", &OccupancyGrid::in_bounds, py::arg("x"), py::arg("y"))
        .def("mark", &OccupancyGrid::mark, py::arg("x"), py::arg("y"),
             py::arg("occupied") = true)
        .def("mark_points", &OccupancyGrid::mark_points, py::arg("points"),
             py::arg("occupied") = true)
        .def("raycast", &OccupancyGrid::raycast, py::arg("ox"), py::arg("oy"),
             py::arg("x"), py::arg("y"), py::arg("hit") = true)
        .def("insert_scan", &OccupancyGrid::insert_scan, py::arg("points"),
             py::arg("origin"), py::arg("max_range") = py::none())
        .def("log_odds", &OccupancyGrid::log_odds, py::arg("x"), py::arg("y"))
        .def("is_occupied", &OccupancyGrid::is_occupied, py::arg("x"),
             py::arg("y"), py::arg("threshold") = 0.6)
        .def("occupied_fraction", &OccupancyGrid::occupied_fraction)
        .def("free_space_ahead", &OccupancyGrid::free_space_ahead,
             py::arg("ego_x"), py::arg("ego_y"), py::arg("yaw"),
             py::arg("max_dist") = 80.0, py::arg("half_width") = 1.0)
        .def("data", &OccupancyGrid::data,
             "Flat row-major log-odds (h*w floats).")
        .def_property_readonly(
            "grid",
            [](const OccupancyGrid& g) {
                py::array_t<float> a({g.h, g.w});
                std::memcpy(a.mutable_data(), g.data().data(),
                            g.data().size() * sizeof(float));
                return a;
            },
            "Copy of the grid as an (h, w) float32 numpy array.")
        .def("as_uint8", [](const OccupancyGrid& g) {
            py::array_t<std::uint8_t> a({g.h, g.w});
            const std::vector<std::uint8_t> v = g.as_uint8();
            std::memcpy(a.mutable_data(), v.data(), v.size());
            return a;
        });
}

#else  // !FSD_WITH_PYBIND

// Compiled standalone (e.g. as part of fsd_core without pybind11):
// intentionally an empty translation unit.
namespace fsd {
namespace detail {
int pybind_module_stub() { return 0; }
}  // namespace detail
}  // namespace fsd

#endif  // FSD_WITH_PYBIND
