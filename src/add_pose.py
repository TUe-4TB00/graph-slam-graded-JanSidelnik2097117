
import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)


def add_pose(graph, initial_estimate):

    # Odometry from x(4) to x(5)

    relative_motion = gtsam.Pose2(
        2.0,          # move forward 2m
        0.0,
        math.pi / 2   # rotate 90deg 
    )

    graph.add(
        gtsam.BetweenFactorPose2(
            X(4), X(5),
            relative_motion,
            ODOMETRY_NOISE
        )
    )

    # Initial estimate for x(5)

    pose4 = initial_estimate.atPose2(X(4))

    pose5_guess = pose4.compose(relative_motion)

    initial_estimate.insert(X(5), pose5_guess)

    return graph, initial_estimate
