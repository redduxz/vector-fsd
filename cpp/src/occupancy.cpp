// Implementation of the BEV occupancy grid — a faithful C++17 port of
// fsd/perception/occupancy.py (numpy arrays -> row-major std::vector<float>).
#include "fsd/occupancy.hpp"

#include <algorithm>
#include <cmath>

namespace fsd {

OccupancyGrid::OccupancyGrid(double width_m, double height_m, double resolution,
                             std::pair<double, double> origin)
    : res(resolution), ox(origin.first), oy(origin.second) {
    w = static_cast<int>(round_nearest(width_m / res));
    h = static_cast<int>(round_nearest(height_m / res));
    grid_.assign(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0.0f);
}

// ------------------------------------------------------------------ basics

void OccupancyGrid::clear() { std::fill(grid_.begin(), grid_.end(), 0.0f); }

void OccupancyGrid::set_origin(double new_ox, double new_oy, bool keep) {
    if (keep && std::abs(new_ox - ox) < w * res &&
        std::abs(new_oy - oy) < h * res) {
        const int dx = static_cast<int>(round_nearest((new_ox - ox) / res));
        const int dy = static_cast<int>(round_nearest((new_oy - oy) / res));
        std::vector<float> next(grid_.size(), 0.0f);
        // Python: new[sy0:sy1, sx0:sx1] = grid[dy0:dy1, dx0:dx1]
        const int sx0 = std::max(0, -dx), sx1 = std::min(w, w - dx);
        const int sy0 = std::max(0, -dy), sy1 = std::min(h, h - dy);
        const int dx0 = std::max(0, dx), dx1 = std::min(w, w + dx);
        const int dy0 = std::max(0, dy), dy1 = std::min(h, h + dy);
        for (int syy = sy0, dyy = dy0; syy < sy1 && dyy < dy1; ++syy, ++dyy) {
            for (int sxx = sx0, dxx = dx0; sxx < sx1 && dxx < dx1; ++sxx, ++dxx) {
                next[static_cast<std::size_t>(syy) * w + sxx] =
                    grid_[static_cast<std::size_t>(dyy) * w + dxx];
            }
        }
        grid_ = std::move(next);
    } else {
        clear();
    }
    ox = new_ox;
    oy = new_oy;
}

std::pair<int, int> OccupancyGrid::cell(double x, double y) const {
    // static_cast<int> truncates toward zero — same as Python int().
    return {static_cast<int>((x - ox) / res),
            static_cast<int>((y - oy) / res)};
}

bool OccupancyGrid::in_bounds(double x, double y) const {
    const auto [cx, cy] = cell(x, y);
    return cx >= 0 && cx < w && cy >= 0 && cy < h;
}

// ------------------------------------------------------------------ update

void OccupancyGrid::bump(int cx, int cy, double delta) {
    if (cx >= 0 && cx < w && cy >= 0 && cy < h) {
        float& v = grid_[static_cast<std::size_t>(cy) * w + cx];
        v = static_cast<float>(
            clampf(v + delta, lo_min, lo_max));
    }
}

void OccupancyGrid::mark(double x, double y, bool occupied) {
    const auto [cx, cy] = cell(x, y);
    bump(cx, cy, occupied ? lo_hit : lo_free);
}

void OccupancyGrid::mark_points(const std::vector<Vec3>& points, bool occupied) {
    for (const Vec3& p : points) mark(p.x, p.y, occupied);
}

void OccupancyGrid::raycast(double rox, double roy, double x, double y,
                            bool hit) {
    auto [cx0, cy0] = cell(rox, roy);
    auto [cx1, cy1] = cell(x, y);
    const int dx = std::abs(cx1 - cx0);
    const int dy = -std::abs(cy1 - cy0);
    const int sx = cx0 < cx1 ? 1 : -1;
    const int sy = cy0 < cy1 ? 1 : -1;
    int err = dx + dy;
    int cx = cx0, cy = cy0;
    while (true) {
        if (cx == cx1 && cy == cy1) {
            bump(cx, cy, hit ? lo_hit : lo_free);
            break;
        }
        bump(cx, cy, lo_free);
        const int e2 = 2 * err;
        if (e2 >= dy) {
            err += dy;
            cx += sx;
        }
        if (e2 <= dx) {
            err += dx;
            cy += sy;
        }
    }
}

void OccupancyGrid::insert_scan(const std::vector<Vec3>& points,
                                std::pair<double, double> origin,
                                std::optional<double> max_range) {
    const double sox = origin.first, soy = origin.second;
    for (const Vec3& p : points) {
        if (max_range &&
            std::hypot(p.x - sox, p.y - soy) > *max_range)
            continue;
        raycast(sox, soy, p.x, p.y, true);
    }
}

// ------------------------------------------------------------------- query

double OccupancyGrid::log_odds(double x, double y) const {
    const auto [cx, cy] = cell(x, y);
    if (cx >= 0 && cx < w && cy >= 0 && cy < h)
        return static_cast<double>(grid_[static_cast<std::size_t>(cy) * w + cx]);
    return 0.0;
}

bool OccupancyGrid::is_occupied(double x, double y, double threshold) const {
    const auto [cx, cy] = cell(x, y);
    return cx >= 0 && cx < w && cy >= 0 && cy < h &&
           grid_[static_cast<std::size_t>(cy) * w + cx] > threshold;
}

double OccupancyGrid::occupied_fraction() const {
    if (grid_.empty()) return 0.0;
    std::size_t count = 0;
    for (float v : grid_)
        if (v > 0.6f) ++count;
    return static_cast<double>(count) / static_cast<double>(grid_.size());
}

double OccupancyGrid::free_space_ahead(double ego_x, double ego_y, double yaw,
                                       double max_dist,
                                       double half_width) const {
    const double ux = std::cos(yaw), uy = std::sin(yaw);
    const double vx = -uy, vy = ux;  // lateral unit vector
    const int n_lat = std::max(1, static_cast<int>(std::ceil(half_width / res)));
    // lats = linspace(-half_width, half_width, 2*n_lat + 1)
    std::vector<double> lats(static_cast<std::size_t>(2 * n_lat + 1));
    for (int i = 0; i <= 2 * n_lat; ++i)
        lats[static_cast<std::size_t>(i)] =
            -half_width + (2.0 * half_width) * i / (2.0 * n_lat);
    const double step = res;
    double s = step;
    while (s < max_dist) {
        const double bx = ego_x + ux * s, by = ego_y + uy * s;
        if (!in_bounds(bx, by)) return s;  // ran out of known space
        for (double lat : lats) {
            if (is_occupied(bx + vx * lat, by + vy * lat)) return s;
        }
        s += step;
    }
    return max_dist;
}

std::vector<std::uint8_t> OccupancyGrid::as_uint8() const {
    std::vector<std::uint8_t> out(grid_.size());
    for (std::size_t i = 0; i < grid_.size(); ++i)
        out[i] = static_cast<std::uint8_t>(
            clampf(128.0 + static_cast<double>(grid_[i]) * 64.0, 0.0, 255.0));
    return out;
}

}  // namespace fsd
