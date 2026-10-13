"""Experimental flat-ground kinematic odometry, with no simulator base position.

Assumes at least one non-slipping support point. This assumption cannot be proven
from joint/IMU feedback alone; hardware use requires contact/slip validation.
"""
import numpy as np


class StanceOdometry:
    def __init__(self):
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.anchors = {}
        self.last_stamp = None
        self.unanchored_seconds = 0.

    def update(self, points, stamp):
        points = np.asarray(points, dtype=float)
        if points.shape != (8, 3) or not np.isfinite(points).all() or not np.isfinite(stamp):
            raise ValueError('Invalid sole kinematics')
        dt = .02 if self.last_stamp is None else stamp-self.last_stamp
        if not 0 < dt <= .04:
            raise ValueError('Stale sole kinematics')
        support = set(np.flatnonzero(points[:, 2] <= points[:, 2].min()+.004).tolist())
        continuing = support.intersection(self.anchors)
        previous = self.position.copy()
        if continuing:
            self.position = np.mean([self.anchors[i]-points[i, :2] for i in continuing], axis=0)
            self.unanchored_seconds = 0.
        elif self.last_stamp is not None:
            self.unanchored_seconds += dt
            if self.unanchored_seconds > .1:
                raise ValueError('No continuous support anchor')
            self.position += self.velocity*dt
        self.anchors = {i: self.anchors[i] if i in continuing else self.position+points[i, :2] for i in support}
        self.velocity = .75*self.velocity + .25*(self.position-previous)/dt
        self.last_stamp = stamp
        return self.position.copy(), self.velocity.copy()
