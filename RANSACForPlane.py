import random
from copy import copy
import numpy as np
from numpy.random import default_rng
from mpl_toolkits import mplot3d
rng = default_rng()


class RANSAC:
    def __init__(self, points = None, n=10, t=0.05,k=1000, d=20):
        self.points = points
        self.n = n
        self.t = t
        self.k = k
        self.d = d

    def fit(self, points):
        best_eq = None
        best_inliers = []
        for _ in range(self.k):
            selections = np.random.choice(len(points), size=3, replace=False)

            basisPoints = points[selections]

            vector1 = basisPoints[1, :]-basisPoints[0, :]
            vector2 = basisPoints[2, :]-basisPoints[0, :]
            vector3 = np.cross(vector1,vector2)

            vector3 = vector3/np.linalg.norm(vector3)

            scalar = -(vector3[0]*basisPoints[1,0]+vector3[1]*basisPoints[1,1]+vector3[2]*basisPoints[1,2])

            plane_eq = [vector3[0],vector3[1],vector3[2],scalar]

            inliers = []

            num = np.abs(plane_eq[0] * points[:, 0] + plane_eq[1] * points[:, 1] + plane_eq[2] * points[:, 2] + plane_eq[3])
            den = np.sqrt(plane_eq[0] ** 2 + plane_eq[1] ** 2 + plane_eq[2] ** 2)
            dist_pt = np.abs(num/den)
            inliers = np.where(dist_pt <= self.t)[0]
            if len(inliers) > len(best_inliers):
                best_eq = plane_eq
                best_inliers=inliers

        return best_eq, best_inliers
