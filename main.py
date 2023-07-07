import sys
import RANSACForPlane as RP
import numpy as np
import random
from mpl_toolkits import mplot3d

if __name__ == "__main__":

    # Sets up the object for calculating plane equation
    regressor = RP.RANSAC()

    pointsxz = np.array([np.random.random_sample(size=70)*10,np.random.random_sample(size=70)/10,np.random.random_sample(size=70)*5]).T
    pointsyz = np.array([np.random.random_sample(size=70)/10,np.random.random_sample(size=70)*10,np.random.random_sample(size=70)*5]).T
    pointsxy = np.array([np.random.random_sample(size=70)*10,np.random.random_sample(size=70)*10,np.random.random_sample(size=70)/10]).T
    points = np.concatenate((pointsxy,pointsyz,pointsxz), axis=0)
    # plane_eq1, inlyingPts1 = regressor.fit(pointsxy)
    # plane_eq2, inlyingPts2 = regressor.fit(pointsyz)
    # plane_eq3, inlyingPts3 = regressor.fit(pointsxz)

    pointID = random.sample(range(len(points)),1)
    point = points[pointID]
    neighborhoodDist = 3

    pointDists = np.sqrt(np.sum((points-point)**2,1))
    pointGroup = np.where(pointDists<=neighborhoodDist)[0]
    pointGroup = points[pointGroup]

    plane_eq1, inlyingPts1 = regressor.fit(pointGroup)

    if plane_eq1 is None:
        sys.exit()

    import matplotlib.pyplot as plt

    x = np.linspace(-1,1,10)
    y = np.linspace(-1,1,10)

    X,Y = np.meshgrid(x,y)
    Z1 = (plane_eq1[0] * X + plane_eq1[1] * Y + plane_eq1[3]) / plane_eq1[2]
    # Z2 = (plane_eq2[0] * X + plane_eq2[1] * Y + plane_eq2[3]) / plane_eq2[2]
    # Z3 = (plane_eq3[0] * X + plane_eq3[1] * Y + plane_eq3[3]) / plane_eq3[2]

    plt.style.use("classic")
    fig, ax = plt.subplots(1,1)
    ax.set_box_aspect(1)
    ax = plt.axes(projection = '3d')
    # ax.scatter3D(points[:,0],points[:,1],points[:,2],cmap='Greens')
    ax.scatter3D(pointGroup[:,0],pointGroup[:,1],pointGroup[:,2],cmap='viridis')
    ax.plot_surface(X, Y, Z1, cmap=plt.get_cmap('cool'), edgecolor='none')
    # ax.plot_surface(X, Y, Z2, cmap=plt.get_cmap('cool'), edgecolor='none')
    # ax.plot_surface(X, Y, Z3, cmap=plt.get_cmap('cool'), edgecolor='none')

    ax.set_xlim3d(-10,10)
    ax.set_ylim3d(-10, 10)
    ax.set_zlim3d(-10, 10)

    plt.show()