"""Route planner — global path from start to goal.

Two modes:

* With a CARLA map (`carla_map` given and the `carla` package importable):
  the lane-level graph from ``map.get_topology()`` is searched with A*
  (euclidean heuristic). Topology pairs are connected road segments; nodes
  are keyed on rounded waypoint location so shared endpoints merge —
  including the junction seams where distinct waypoint objects coincide.
  The winning segment chain is then densified with
  ``waypoint.next_until_lane_end`` (with a ``.next()`` walk as fallback).

* Without CARLA: straight-line interpolation at `resolution` spacing —
  good enough for unit tests and open lots.
"""
from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Tuple

from fsd.core.logger import get
from fsd.core.types import Vec3, Waypoint

try:                                   # CARLA is optional at import time
    import carla                        # type: ignore
    _HAS_CARLA = True
except Exception:                      # pragma: no cover - no CARLA env
    carla = None                       # type: ignore
    _HAS_CARLA = False

_log = get("planning.route")

_Node = Tuple[float, float, float]


def _node_key(loc) -> _Node:
    """Round a CARLA Location to ~0.1 m so coincident waypoints merge."""
    return (round(loc.x, 1), round(loc.y, 1), round(loc.z, 1))


class RoutePlanner:
    """Global route over the road graph (or straight interpolation)."""

    def __init__(
        self,
        resolution_m: float = 2.0,
        default_speed_limit_mps: float = 13.9,
        max_expand: int = 200_000,
        max_segment_points: int = 400,
    ) -> None:
        self.resolution = float(resolution_m)
        self.speed_limit = float(default_speed_limit_mps)
        self.max_expand = int(max_expand)
        self.max_segment_points = int(max_segment_points)

    # ------------------------------------------------------------------ #
    def plan(
        self,
        start,
        goal,
        carla_map=None,
    ) -> List[Waypoint]:
        """Return a dense waypoint list from `start` to `goal`.

        `start`/`goal` accept Vec3, Waypoint or any object with x/y/z.
        """
        sx, sy, sz = float(start.x), float(start.y), float(start.z or 0.0)
        gx, gy, gz = float(goal.x), float(goal.y), float(goal.z or 0.0)

        if carla_map is not None and _HAS_CARLA:
            try:
                route = self._plan_carla(sx, sy, sz, gx, gy, gz, carla_map)
                if route:
                    return route
                _log.warning("CARLA A* found no route — straight fallback")
            except Exception as exc:                    # pragma: no cover
                _log.warning("CARLA route failed (%s) — straight fallback",
                             exc)
        return self._plan_straight(sx, sy, sz, gx, gy, gz)

    # ------------------------------------------------------------------ #
    # straight-line fallback                                              #
    # ------------------------------------------------------------------ #
    def _plan_straight(
        self,
        sx: float, sy: float, sz: float,
        gx: float, gy: float, gz: float,
    ) -> List[Waypoint]:
        dist = math.hypot(gx - sx, gy - sy)
        n = max(int(dist / self.resolution) + 1, 2)
        yaw = math.atan2(gy - sy, gx - sx) if dist > 1e-6 else 0.0
        pts = []
        for i in range(n):
            t = i / (n - 1)
            pts.append(Waypoint(
                x=sx + t * (gx - sx),
                y=sy + t * (gy - sy),
                z=sz + t * (gz - sz),
                yaw=yaw,
                speed_limit=self.speed_limit,
            ))
        return pts

    # ------------------------------------------------------------------ #
    # CARLA map A*                                                        #
    # ------------------------------------------------------------------ #
    def _plan_carla(
        self,
        sx: float, sy: float, sz: float,
        gx: float, gy: float, gz: float,
        carla_map,
    ) -> List[Waypoint]:
        topo = carla_map.get_topology()      # List[(Waypoint, Waypoint)]
        if not topo:
            return []

        # Build directed graph: node = rounded location key.
        nodes: Dict[_Node, object] = {}
        adj: Dict[_Node, List[Tuple[_Node, float, object, object]]] = {}
        for wa, wb in topo:
            la, lb = wa.transform.location, wb.transform.location
            ka, kb = _node_key(la), _node_key(lb)
            nodes.setdefault(ka, wa)
            nodes.setdefault(kb, wb)
            w = la.distance(lb)
            adj.setdefault(ka, []).append((kb, w, wa, wb))

        # nearest graph node to start / goal
        k_start = min(nodes, key=lambda k: (k[0] - sx) ** 2 + (k[1] - sy) ** 2)
        k_goal = min(nodes, key=lambda k: (k[0] - gx) ** 2 + (k[1] - gy) ** 2)

        came = self._astar(adj, k_start, k_goal)
        if came is None:
            return []

        # Reconstruct the segment (waypoint-pair) chain.
        edges: List[Tuple[object, object]] = []
        k = k_goal
        while k != k_start:
            prev = came[k]
            if prev is None:
                return []
            k_prev, wa, wb = prev
            edges.append((wa, wb))
            k = k_prev
        edges.reverse()

        # Densify each segment to `resolution` spacing.
        route: List[Waypoint] = []
        for wa, wb in edges:
            route.extend(self._densify_segment(wa, wb))

        # Prepend a link from `start` if the graph entry sits away from it.
        if route:
            first = route[0]
            gap = math.hypot(first.x - sx, first.y - sy)
            if gap > self.resolution:
                yaw0 = math.atan2(first.y - sy, first.x - sx)
                n = max(int(gap / self.resolution), 2)
                link = [Waypoint(
                    x=sx + (i / n) * (first.x - sx),
                    y=sy + (i / n) * (first.y - sy),
                    z=sz + (i / n) * (first.z - sz),
                    yaw=yaw0,
                    speed_limit=self.speed_limit) for i in range(n)]
                route = link + route

        # Tail into the exact goal.
        if route:
            last = route[-1]
            gap = math.hypot(gx - last.x, gy - last.y)
            if gap > self.resolution:
                yaw = math.atan2(gy - last.y, gx - last.x)
                n = max(int(gap / self.resolution), 2)
                for i in range(1, n + 1):
                    t = i / n
                    route.append(Waypoint(
                        x=last.x + t * (gx - last.x),
                        y=last.y + t * (gy - last.y),
                        z=last.z + t * (gz - last.z),
                        yaw=yaw,
                        speed_limit=self.speed_limit))
            else:
                route.append(Waypoint(x=gx, y=gy, z=gz,
                                      yaw=last.yaw,
                                      speed_limit=self.speed_limit))
        return route

    # ------------------------------------------------------------------ #
    def _astar(
        self,
        adj: Dict[_Node, List[Tuple[_Node, float, object, object]]],
        k_start: _Node,
        k_goal: _Node,
    ) -> Optional[Dict[_Node, Tuple[_Node, object, object]]]:
        """A* over the topology graph. Returns came_from map or None."""
        def h(k: _Node) -> float:
            return math.hypot(k[0] - k_goal[0], k[1] - k_goal[1])

        g = {k_start: 0.0}
        came: Dict[_Node, Tuple[_Node, object, object] | None] = {
            k_start: None}
        pq: List[Tuple[float, _Node]] = [(h(k_start), k_start)]
        expanded = 0

        while pq and expanded < self.max_expand:
            _, k = heapq.heappop(pq)
            if k == k_goal:
                return came
            expanded += 1
            for k_next, w, wa, wb in adj.get(k, ()):
                ng = g[k] + w
                if ng < g.get(k_next, math.inf):
                    g[k_next] = ng
                    came[k_next] = (k, wa, wb)
                    heapq.heappush(pq, (ng + h(k_next), k_next))
        _log.warning("A* exhausted (%d expansions)", expanded)
        return None

    # ------------------------------------------------------------------ #
    def _densify_segment(self, wa, wb) -> List[Waypoint]:
        """Convert a topology segment (wa -> wb) to dense Waypoints."""
        out: List[Waypoint] = []
        try:
            chain = wa.next_until_lane_end(self.resolution)
        except Exception:
            chain = self._walk_next(wa, wb)

        end = wb.transform.location
        seg_len = max(wa.transform.location.distance(end), self.resolution)

        acc = 0.0
        prev_loc = wa.transform.location
        out.append(self._convert(wa))
        for cw in chain[: self.max_segment_points]:
            loc = cw.transform.location
            acc += prev_loc.distance(loc)
            prev_loc = loc
            out.append(self._convert(cw))
            if acc >= seg_len - self.resolution * 0.5:
                break
        out.append(self._convert(wb))
        return out

    def _walk_next(self, wa, wb) -> list:
        """Fallback densifier: walk `.next()` toward wb, nearest candidate."""
        chain = []
        cur = wa
        end = wb.transform.location
        for _ in range(self.max_segment_points):
            nxt = cur.next(self.resolution)
            if not nxt:
                break
            cur = min(nxt, key=lambda w: w.transform.location.distance(end))
            chain.append(cur)
            if cur.transform.location.distance(end) < self.resolution:
                break
        return chain

    @staticmethod
    def _convert(wp) -> Waypoint:
        t = wp.transform
        return Waypoint(
            x=t.location.x, y=t.location.y, z=t.location.z,
            yaw=math.radians(t.rotation.yaw))
