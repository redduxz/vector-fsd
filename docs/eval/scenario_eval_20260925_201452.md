# Closed-Loop Scenario Evaluation

_Generated 2026-09-25 20:14:52 — fsd.scenarios.runner (smoke world, no CARLA)_

## Summary

| Scenario | Verdict | Ticks | Dist (m) | min TTC (s) | min gap (m) | jerk RMS | lane RMSE | safe stops |
|---|---|---|---|---|---|---|---|---|
| lead_vehicle_cut_in | PASS | 134 | 44.6 | 0.75 | 2.2 | 22.78 | 0.000 | 1 |
| sudden_braking | PASS | 216 | 77.9 | 1.81 | 5.5 | 13.91 | 0.000 | 0 |
| jaywalker_pedestrian | PASS | 190 | 39.7 | 2.33 | 5.5 | 28.26 | 0.000 | 1 |
| red_light_runner | PASS | 186 | 48.7 | 36.26 | 7.7 | 11.55 | 0.000 | 1 |
| sensor_dropout | PASS | 92 | 29.0 | 46.13 | 55.9 | 23.85 | 0.000 | 1 |
| planner_stall | PASS | 63 | 14.5 | - | 76.0 | 28.08 | 0.000 | 1 |

**Totals:** 6 PASS / 0 FAIL / 0 TIMEOUT

## Details

### lead_vehicle_cut_in — PASS

> Lead vehicle cuts into the ego lane at ~12 m while ego cruises; expect emergency brake / SAFE_STOP, no contact.

| metric | value |
|---|---|
| ticks | 134 |
| sim time | 6.7 s |
| distance | 44.6 m |
| speed avg/max | 6.7 / 12.7 m/s |
| min TTC | 0.75 s |
| min bumper gap | 2.16 m |
| collision | no |
| jerk rms/max | 22.78 / 213.08 m/s^3 |
| max |accel| | 8.25 m/s^2 |
| lane-center RMSE | 0.000 m (134 samples) |
| safe stops / degraded / disengaged | 1 / 0 / 0 |
| crashed ticks | 0 |
| perception/planning failures | 0 / 0 |
| time in mode | ENGAGED 5.1s, SAFE_STOP 1.7s |
| rule hits | free_space:c22,w11; monitor:i1; ttc:c18,w2 |

**Scenario events**

- [t=  5.05] cut_in: lead merges 12 m ahead at 4.0 m/s (ego 12.6 m/s)

**Safety events (deduped)**

- `[critical] ttc: TTC 0.80s below floor 1.50s (obj#1 vehicle gap=7.0m closing=8.7m/s conf=0.95)`
- `[warning] free_space: free space 11.6m nearing minimum 8.0m`
- `[critical] ttc: TTC 0.79s below floor 1.50s (obj#1 vehicle gap=6.5m closing=8.3m/s conf=0.95)`
- `[warning] free_space: free space 11.1m nearing minimum 8.0m`
- `[critical] ttc: TTC 0.78s below floor 1.50s (obj#1 vehicle gap=6.1m closing=7.9m/s conf=0.95)`
- `[warning] free_space: free space 10.7m nearing minimum 8.0m`
- `[critical] ttc: TTC 0.77s below floor 1.50s (obj#1 vehicle gap=5.7m closing=7.4m/s conf=0.95)`
- `[warning] free_space: free space 10.3m nearing minimum 8.0m`
- `[critical] ttc: TTC 0.76s below floor 1.50s (obj#1 vehicle gap=5.4m closing=7.0m/s conf=0.95)`
- `[warning] free_space: free space 10.0m nearing minimum 8.0m`
- `[critical] ttc: TTC 0.76s below floor 1.50s (obj#1 vehicle gap=5.0m closing=6.6m/s conf=0.95)`
- `[warning] free_space: free space 9.6m nearing minimum 8.0m`
- `[critical] ttc: TTC 0.75s below floor 1.50s (obj#1 vehicle gap=4.7m closing=6.2m/s conf=0.95)`
- `[warning] free_space: free space 9.3m nearing minimum 8.0m`
- `[critical] ttc: TTC 0.75s below floor 1.50s (obj#1 vehicle gap=4.4m closing=5.8m/s conf=0.95)`
- … +28 more

### sudden_braking — PASS

> Lead vehicle brakes to zero at ~6 m/s^2 in front of cruising ego; expect controlled emergency stop.

| metric | value |
|---|---|
| ticks | 216 |
| sim time | 10.8 s |
| distance | 77.9 m |
| speed avg/max | 7.2 / 14.3 m/s |
| min TTC | 1.81 s |
| min bumper gap | 5.51 m |
| collision | no |
| jerk rms/max | 13.91 / 165.69 m/s^3 |
| max |accel| | 8.29 m/s^2 |
| lane-center RMSE | 0.000 m (216 samples) |
| safe stops / degraded / disengaged | 0 / 1 / 0 |
| crashed ticks | 0 |
| perception/planning failures | 0 / 0 |
| time in mode | DEGRADED 4.2s, ENGAGED 6.6s |
| rule hits | free_space:w39; ttc:w53 |

**Scenario events**

- [t=  6.05] brake: lead brakes at 6.0 m/s^2 (gap 32.9 m)
- [t=  7.15] note: lead at standstill

**Safety events (deduped)**

- `[warning] ttc: TTC 2.58s approaching floor 1.50s (obj#1 vehicle gap=27.8m closing=10.8m/s conf=0.95)`
- `[warning] ttc: TTC 2.46s approaching floor 1.50s (obj#1 vehicle gap=27.2m closing=11.1m/s conf=0.95)`
- `[warning] ttc: TTC 2.34s approaching floor 1.50s (obj#1 vehicle gap=26.7m closing=11.4m/s conf=0.95)`
- `[warning] ttc: TTC 2.23s approaching floor 1.50s (obj#1 vehicle gap=26.1m closing=11.7m/s conf=0.95)`
- `[warning] ttc: TTC 2.13s approaching floor 1.50s (obj#1 vehicle gap=25.5m closing=12.0m/s conf=0.95)`
- `[warning] ttc: TTC 2.03s approaching floor 1.50s (obj#1 vehicle gap=24.9m closing=12.3m/s conf=0.95)`
- `[warning] ttc: TTC 1.99s approaching floor 1.50s (obj#1 vehicle gap=24.2m closing=12.2m/s conf=0.95)`
- `[warning] ttc: TTC 1.96s approaching floor 1.50s (obj#1 vehicle gap=23.6m closing=12.0m/s conf=0.95)`
- `[warning] ttc: TTC 1.93s approaching floor 1.50s (obj#1 vehicle gap=23.0m closing=11.9m/s conf=0.95)`
- `[warning] ttc: TTC 1.89s approaching floor 1.50s (obj#1 vehicle gap=22.4m closing=11.8m/s conf=0.95)`
- `[warning] ttc: TTC 1.86s approaching floor 1.50s (obj#1 vehicle gap=21.8m closing=11.7m/s conf=0.95)`
- `[warning] ttc: TTC 1.83s approaching floor 1.50s (obj#1 vehicle gap=21.2m closing=11.6m/s conf=0.95)`
- `[warning] ttc: TTC 1.81s approaching floor 1.50s (obj#1 vehicle gap=20.6m closing=11.4m/s conf=0.95)`
- `[warning] ttc: TTC 1.82s approaching floor 1.50s (obj#1 vehicle gap=20.0m closing=11.0m/s conf=0.95)`
- `[warning] ttc: TTC 1.84s approaching floor 1.50s (obj#1 vehicle gap=19.5m closing=10.6m/s conf=0.95)`
- … +69 more

### jaywalker_pedestrian — PASS

> Pedestrian crosses the lane mid-block at ~1 m/s; expect yield/emergency stop, no contact.

| metric | value |
|---|---|
| ticks | 190 |
| sim time | 9.5 s |
| distance | 39.7 m |
| speed avg/max | 4.2 / 10.0 m/s |
| min TTC | 2.33 s |
| min bumper gap | 5.48 m |
| collision | no |
| jerk rms/max | 28.26 / 240.08 m/s^3 |
| max |accel| | 8.20 m/s^2 |
| lane-center RMSE | 0.000 m (190 samples) |
| safe stops / degraded / disengaged | 1 / 1 / 0 |
| crashed ticks | 0 |
| perception/planning failures | 0 / 0 |
| time in mode | DEGRADED 4.1s, ENGAGED 4.5s, SAFE_STOP 1.0s |
| rule hits | free_space:c7,w58; monitor:i1; ttc:w53 |

**Scenario events**

- [t=  3.05] inject: pedestrian steps off at 34 m ahead, crossing at 0.9 m/s
- [t=  9.45] clear: pedestrian cleared the roadway

**Safety events (deduped)**

- `[warning] ttc: TTC 2.40s approaching floor 1.50s (obj#7000 pedestrian gap=18.4m closing=7.7m/s conf=0.88)`
- `[warning] ttc: TTC 2.40s approaching floor 1.50s (obj#7000 pedestrian gap=18.1m closing=7.5m/s conf=0.88)`
- `[warning] ttc: TTC 2.39s approaching floor 1.50s (obj#7000 pedestrian gap=17.7m closing=7.4m/s conf=0.88)`
- `[warning] ttc: TTC 2.39s approaching floor 1.50s (obj#7000 pedestrian gap=17.3m closing=7.3m/s conf=0.88)`
- `[warning] ttc: TTC 2.39s approaching floor 1.50s (obj#7000 pedestrian gap=17.0m closing=7.1m/s conf=0.88)`
- `[warning] ttc: TTC 2.38s approaching floor 1.50s (obj#7000 pedestrian gap=16.6m closing=7.0m/s conf=0.88)`
- `[warning] ttc: TTC 2.38s approaching floor 1.50s (obj#7000 pedestrian gap=16.3m closing=6.8m/s conf=0.88)`
- `[warning] ttc: TTC 2.38s approaching floor 1.50s (obj#7000 pedestrian gap=16.0m closing=6.7m/s conf=0.88)`
- `[warning] ttc: TTC 2.37s approaching floor 1.50s (obj#7000 pedestrian gap=15.6m closing=6.6m/s conf=0.88)`
- `[warning] ttc: TTC 2.37s approaching floor 1.50s (obj#7000 pedestrian gap=15.3m closing=6.5m/s conf=0.88)`
- `[warning] ttc: TTC 2.37s approaching floor 1.50s (obj#7000 pedestrian gap=15.0m closing=6.3m/s conf=0.88)`
- `[warning] ttc: TTC 2.37s approaching floor 1.50s (obj#7000 pedestrian gap=14.7m closing=6.2m/s conf=0.88)`
- `[warning] ttc: TTC 2.36s approaching floor 1.50s (obj#7000 pedestrian gap=14.4m closing=6.1m/s conf=0.88)`
- `[warning] ttc: TTC 2.36s approaching floor 1.50s (obj#7000 pedestrian gap=14.1m closing=6.0m/s conf=0.88)`
- `[warning] ttc: TTC 2.35s approaching floor 1.50s (obj#7000 pedestrian gap=13.8m closing=5.9m/s conf=0.88)`
- … +78 more

### red_light_runner — PASS

> Light goes yellow->red ahead; a cross-traffic vehicle runs the intersection. Expect a stop at the line, no red-light violation, no contact.

| metric | value |
|---|---|
| ticks | 186 |
| sim time | 9.3 s |
| distance | 48.7 m |
| speed avg/max | 5.2 / 11.0 m/s |
| min TTC | 36.26 s |
| min bumper gap | 7.70 m |
| collision | no |
| jerk rms/max | 11.55 / 84.95 m/s^3 |
| max |accel| | 8.10 m/s^2 |
| lane-center RMSE | 0.000 m (186 samples) |
| safe stops / degraded / disengaged | 1 / 1 / 0 |
| crashed ticks | 0 |
| perception/planning failures | 0 / 0 |
| time in mode | DEGRADED 0.9s, ENGAGED 5.8s, SAFE_STOP 2.6s |
| rule hits | free_space:c52,w18; monitor:i1 |

**Scenario events**

- [t=  2.55] trigger: light -> YELLOW
- [t=  3.55] trigger: light -> RED
- [t=  6.05] inject: cross-traffic vehicle enters intersection

**Safety events (deduped)**

- `[warning] free_space: free space 13.9m nearing minimum 8.0m`
- `[warning] free_space: free space 13.5m nearing minimum 8.0m`
- `[warning] free_space: free space 13.1m nearing minimum 8.0m`
- `[warning] free_space: free space 12.7m nearing minimum 8.0m`
- `[warning] free_space: free space 12.3m nearing minimum 8.0m`
- `[warning] free_space: free space 11.9m nearing minimum 8.0m`
- `[warning] free_space: free space 11.5m nearing minimum 8.0m`
- `[warning] free_space: free space 11.1m nearing minimum 8.0m`
- `[warning] free_space: free space 10.8m nearing minimum 8.0m`
- `[warning] free_space: free space 10.4m nearing minimum 8.0m`
- `[warning] free_space: free space 10.1m nearing minimum 8.0m`
- `[warning] free_space: free space 9.8m nearing minimum 8.0m`
- `[warning] free_space: free space 9.4m nearing minimum 8.0m`
- `[warning] free_space: free space 9.1m nearing minimum 8.0m`
- `[warning] free_space: free space 8.8m nearing minimum 8.0m`
- … +14 more

### sensor_dropout — PASS

> Object-detection feed drops for a window; watchdog must flag stale perception and veto to SAFE_STOP.

| metric | value |
|---|---|
| ticks | 92 |
| sim time | 4.6 s |
| distance | 29.0 m |
| speed avg/max | 6.3 / 11.5 m/s |
| min TTC | 46.13 s |
| min bumper gap | 55.90 m |
| collision | no |
| jerk rms/max | 23.85 / 212.46 m/s^3 |
| max |accel| | 8.23 m/s^2 |
| lane-center RMSE | 0.000 m (92 samples) |
| safe stops / degraded / disengaged | 1 / 0 / 0 |
| crashed ticks | 0 |
| perception/planning failures | 11 / 0 |
| time in mode | ENGAGED 4.5s, SAFE_STOP 0.1s |
| rule hits | monitor:i1; watchdog:c2 |

**Scenario events**

- [t=  4.05] inject: objects feed dropped at ego 10.3 m/s
- [t=  4.55] note: SAFE_STOP engaged
- [t=  4.55] note: stage failures: perceptionx11

**Safety events (deduped)**

- `[critical] watchdog: perception output 0.52s stale (timeout 0.50s)`
- `[critical] watchdog: perception output 0.56s stale (timeout 0.50s)`

### planner_stall — PASS

> Trajectory stage heartbeat lost mid-drive; expect immediate safe-stop command + watchdog confirmation.

| metric | value |
|---|---|
| ticks | 63 |
| sim time | 3.2 s |
| distance | 14.5 m |
| speed avg/max | 4.6 / 8.2 m/s |
| min TTC | - |
| min bumper gap | 76.00 m |
| collision | no |
| jerk rms/max | 28.08 / 205.59 m/s^3 |
| max |accel| | 8.16 m/s^2 |
| lane-center RMSE | 0.000 m (63 samples) |
| safe stops / degraded / disengaged | 1 / 0 / 0 |
| crashed ticks | 0 |
| perception/planning failures | 0 / 2 |
| time in mode | ENGAGED 3.1s, SAFE_STOP 0.1s |
| rule hits | monitor:i1 |

**Scenario events**

- [t=  3.05] inject: planner heartbeat lost — stage stalled
- [t=  3.10] note: SAFE_STOP engaged
- [t=  3.10] note: stage failures: planningx2

