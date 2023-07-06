import random
from copy import copy
import numpy as np
from numpy.random import default_rng
from mpl_toolkits import mplot3d
rng = default_rng()


class RANSAC:
    def __init__(self, points = None, n=10, t=0.05,k=100, d = 10):
        self.points = points
        self.n = n
        self.t = t
        self.k = k
        self.d = d

    def fit(self, points):
        best_eq = None

        for _ in range(self.k):
            selections = np.random.choice(len(points), size=3, replace=False)

            basisPoints = points[selections]

            vector1 = basisPoints[1]-basisPoints[0]
            vector2 = basisPoints[2]-basisPoints[0]
            vector3 = np.cross(vector1,vector2)
            vector3 = vector3/np.linalg.norm(vector3)

            scalar = -(vector3[0]*basisPoints[0,0]+vector3[1]*basisPoints[0,1]+vector3[2]*basisPoints[0,2])

            plane_eq = np.array([vector3[0],vector3[1],vector3[2],scalar])

            num = np.abs(vector3[0] * points[:, 0] + vector3[1] * points[:, 1] + vector3[2] * points[:, 2] + scalar)
            den = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2)
            dist_pt = np.abs(num/den)

            inliers = np.where(dist_pt >= self.t)[0]
            print(len(inliers))
            if len(inliers) > self.d:
                best_eq = plane_eq
            # for it in points:
            #     num = np.abs(vector3[0]*it[0]+ vector3[1]*it[1]+vector3[2]*it[2]+scalar)
            #     den = np.sqrt(it[0]**2+it[1]**2+it[2]**2)
            #     if num/den > self.t:
            #         basisPoints[len(basisPoints)] = it
        return best_eq
