// Bird's-eye occupancy grid (world frame, log-odds) — C++17 port of
// fsd/perception/occupancy.py.
//
// Cells accumulate log-odds evidence: raycast cells get free evidence,
// endpoints get occupied evidence. Positive log-odds = occupied. The grid is
// anchored at a world origin that can be recentered on the ego as it moves.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "fsd/types.hpp"

namespace fsd {

/// 2-D BEV occupancy grid; cell (0,0) sits at world (ox, oy).
/// Storage is row-major [cy * w + cx], float log-odds.
class OccupancyGrid {
public:
    double res;             ///< m/cell
    int w = 0, h = 0;       ///< cells
    double ox = 0.0, oy = 0.0;  ///< world coords of cell (0,0)
    double lo_hit = 0.9, lo_free = -0.4;
    double lo_min = -4.0, lo_max = 4.0;

    explicit OccupancyGrid(double width_m = 120.0, double height_m = 120.0,
                           double resolution = 0.25,
                           std::pair<double, double> origin = {0.0, 0.0});

    // --------------------------------------------------------------- basics
    void clear();

    /// Move the world anchor (e.g. recenter on ego); optionally keep data
    /// by shifting the evidence that still overlaps the new window.
    void set_origin(double new_ox, double new_oy, bool keep = true);

    /// Cell indices for a world point (truncation semantics as Python int()).
    std::pair<int, int> cell(double x, double y) const;
    bool in_bounds(double x, double y) const;

    // --------------------------------------------------------------- update
    /// Add occupied/free evidence for the cell containing world (x, y).
    void mark(double x, double y, bool occupied = true);
    void mark_points(const std::vector<Vec3>& points, bool occupied = true);

    /// Bresenham from (rox,roy) to (x,y); free evidence along the ray,
    /// occupied evidence at the endpoint if `hit`.
    void raycast(double rox, double roy, double x, double y, bool hit = true);

    /// Raycast each world-frame scan endpoint from the sensor origin.
    void insert_scan(const std::vector<Vec3>& points,
                     std::pair<double, double> origin,
                     std::optional<double> max_range = std::nullopt);

    // ---------------------------------------------------------------- query
    double log_odds(double x, double y) const;
    bool is_occupied(double x, double y, double threshold = 0.6) const;
    double occupied_fraction() const;

    /// Meters of clear corridor along `yaw` before an occupied cell.
    ///
    /// Marches a corridor of 2*half_width lateral clearance; the first
    /// occupied cell across the corridor wins. Leaving the grid bounds is
    /// reported as the distance to the grid edge (unknown beyond).
    double free_space_ahead(double ego_x, double ego_y, double yaw,
                            double max_dist = 80.0,
                            double half_width = 1.0) const;

    /// 0 free / 128 unknown / 255 occupied — handy for debugging dumps.
    std::vector<std::uint8_t> as_uint8() const;

    /// Raw row-major log-odds storage, h*w floats.
    const std::vector<float>& data() const { return grid_; }

private:
    void bump(int cx, int cy, double delta);

    std::vector<float> grid_;
};

}  // namespace fsd
