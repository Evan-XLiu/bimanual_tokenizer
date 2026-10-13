"""Bounded outer position loop for the pinned velocity policy.

Requires an independently supplied, fresh world pose and velocity. LowState alone
does NOT provide these. Simulator truth is allowed only as a labelled sim source.
"""
import math
import numpy as np

from lower_body_policy import finite


class StandingReference:
    def __init__(self, position, yaw, stamp):
        self.anchor = finite(position, 2).astype(float)
        if not all(math.isfinite(x) for x in (yaw, stamp)):
            raise ValueError('Invalid standing anchor')
        self.yaw = yaw
        self.last_stamp = stamp
        self.integral = np.zeros(2)

    def update(self, position, velocity, yaw, stamp, now):
        if not all(math.isfinite(x) for x in (yaw, stamp, now)) or not 0 <= now-stamp <= .04:
            raise ValueError('Stale or invalid world pose')
        dt = stamp-self.last_stamp
        if not 0 < dt <= .04:
            raise ValueError('World pose must advance at least every 40ms')
        error = finite(position, 2)-self.anchor
        velocity = finite(velocity, 2)
        candidate = np.clip(self.integral+error*dt, -.2, .2)
        world = -1.5*error-.5*velocity-candidate
        rotate = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])
        body = rotate@world
        # Freeze integral during command saturation (bounded anti-windup).
        if np.max(np.abs(body)) <= .3:
            self.integral = candidate
        heading_error = math.atan2(math.sin(yaw-self.yaw), math.cos(yaw-self.yaw))
        self.last_stamp = stamp
        return np.r_[np.clip(body, -.3, .3), np.clip(-heading_error, -.4, .4)]


def quaternion_yaw(quat):
    w, x, y, z = finite(quat, 4)
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
