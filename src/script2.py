import numpy as np
import matplotlib.pyplot as plt
import skimage
import scipy.ndimage
import math
import cv2

from cp_hw6 import pixel2ray, set_axes_equal

verticalTopLeft = (0, 255)
verticalBottomRight = (320, 795)
horizontalTopLeft = (655, 200)
horizontalBottomRight = (767, 835)
shapeVertical = (verticalBottomRight[0] - verticalTopLeft[0] + 1, verticalBottomRight[1] - verticalTopLeft[1] + 1)
shapeHorizontal = (horizontalBottomRight[0] - horizontalTopLeft[0] + 1, horizontalBottomRight[1] - horizontalTopLeft[1] + 1)

startFrame = 56
endFrame = 150
numFrames = endFrame - startFrame + 1

with np.load('shadowEdges.npz') as X:
    edgeVertical, edgeHorizontal = [X[i] for i in ('edgeVertical', 'edgeHorizontal')]
with np.load('shadowTime.npz') as X:
    shadowTime = X['shadowTime']
with np.load('intrinsic_calib.npz') as X:
    K, distortion = [X[i] for i in ('mtx', 'dist')]
with np.load('extrinsic_calib.npz') as X:
    T_h, R_h, T_v, R_v = [X[i] for i in ('tvec_h', 'rmat_h', 'tvec_v', 'rmat_v')]
with np.load('intensityDiff.npz') as X:
    I_diff = X['I_diff']

imageShape = shadowTime.shape

image0 = plt.imread('../data/frog/000001.jpg')
image0 = np.divide(image0, 255.0, dtype=np.float32)
image0 = cv2.cvtColor(image0, cv2.COLOR_RGB2XYZ)
image0 = image0[:, :, 1]
# plt.imshow(image0, cmap='gray')
# plt.show()


# ray1s = np.zeros((numFrames, 3))
# ray2s = np.zeros((numFrames, 3))
# ray3s = np.zeros((numFrames, 3))
# ray4s = np.zeros((numFrames, 3))

# fig = plt.figure()
# ax = plt.axes()

## Calibration of shadow lines
pts_3d = np.zeros((numFrames, 4, 3))
for i in range(startFrame, endFrame + 1) :
    p1_2d = np.array([0, edgeVertical[i, 2] / (-edgeVertical[i, 1])])
    p2_2d = np.array([450, (450 * edgeVertical[i, 0] + edgeVertical[i, 2]) / (-edgeVertical[i, 1])])
    p3_2d = np.array([450, (450 * edgeHorizontal[i, 0] + edgeHorizontal[i, 2]) / (-edgeHorizontal[i, 1])])
    p4_2d = np.array([imageShape[0], (imageShape[0] * edgeHorizontal[i, 0] + edgeHorizontal[i, 2]) / (-edgeHorizontal[i, 1])])
    # plt.plot([p1_2d[1], p2_2d[1]], [p1_2d[0], p2_2d[0]], color='red')
    # plt.plot([p3_2d[1], p4_2d[1]], [p3_2d[0], p4_2d[0]], color='green')
    # plt.imshow(image0, cmap='gray')
    # plt.show()
    ray1_camera = pixel2ray(p1_2d, K, distortion).reshape((3, 1))
    ray2_camera = pixel2ray(p2_2d, K, distortion).reshape((3, 1))
    ray3_camera = pixel2ray(p3_2d, K, distortion).reshape((3, 1))
    ray4_camera = pixel2ray(p4_2d, K, distortion).reshape((3, 1))
    # ray1s[i-startFrame] = ray1_camera.flatten()
    # ray2s[i-startFrame] = ray2_camera.flatten()
    # ray3s[i-startFrame] = ray3_camera.flatten()
    # ray4s[i-startFrame] = ray4_camera.flatten()
    camera_vertical = R_v.T @ (-T_v)
    camera_horizontal = R_h.T @ (-T_h)
    ray1_vertical = R_v.T @ ray1_camera
    ray2_vertical = R_v.T @ ray2_camera
    ray3_horizontal = R_h.T @ ray3_camera
    ray4_horizontal = R_h.T @ ray4_camera
    t1 = camera_vertical[2] / ray1_vertical[2]
    t2 = camera_vertical[2] / ray2_vertical[2]
    t3 = camera_horizontal[2] / ray3_horizontal[2]
    t4 = camera_horizontal[2] / ray4_horizontal[2]
    p1_vertical = camera_vertical - t1 * ray1_vertical
    p2_vertical = camera_vertical - t2 * ray2_vertical
    p3_horizontal = camera_horizontal - t3 * ray3_horizontal
    p4_horizontal = camera_horizontal - t4 * ray4_horizontal
    p1_camera = R_v @ p1_vertical + T_v
    p2_camera = R_v @ p2_vertical + T_v
    p3_camera = R_h @ p3_horizontal + T_h
    p4_camera = R_h @ p4_horizontal + T_h
    pts_3d[i-startFrame, 0] = p1_camera.flatten()
    pts_3d[i-startFrame, 1] = p2_camera.flatten()
    pts_3d[i-startFrame, 2] = p3_camera.flatten()
    pts_3d[i-startFrame, 3] = p4_camera.flatten()


# plt.imshow(image0, cmap='gray')
# plt.show()

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter(ray1s[:, 0].flatten(), ray1s[:, 0].flatten(), ray1s[:, 0].flatten(), color='red')
# ax.scatter(ray2s[:, 0].flatten(), ray2s[:, 0].flatten(), ray2s[:, 0].flatten(), color='orange')
# ax.scatter(ray3s[:, 0].flatten(), ray3s[:, 0].flatten(), ray3s[:, 0].flatten(), color='green')
# ax.scatter(ray4s[:, 0].flatten(), ray4s[:, 0].flatten(), ray4s[:, 0].flatten(), color='blue')
# ax.scatter(0, 0, 0, color='black')
# plt.show()

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# for i in range(numFrames) :
#     for j in range(4) :
#         if j < 2 :
#             c = 'red'
#         else :
#             c = 'green'
#         ax.scatter(pts_3d[i, j, 0], pts_3d[i, j, 1], pts_3d[i, j, 2], color=c)
# plt.show()


np.savez('shadowlinePoints3D.npz', pts_3d=pts_3d)

planePoints = pts_3d[:, 0, :].reshape(-1, 3)
planeNormals = np.zeros(planePoints.shape)
for i in range(planePoints.shape[0]) :
    planeNormals[i] = np.cross(pts_3d[i, 1] - pts_3d[i, 0], pts_3d[i, 3] - pts_3d[i, 2])
    planeNormals[i] = planeNormals[i] / np.linalg.norm(planeNormals[i])
np.savez('shadowPlane.npz', points=planePoints, normals=planeNormals)


## Reconstruction
cropArea = [[325, 310], [625, 810]]
shadowTime = shadowTime - startFrame
reconstructedPoints = [[], [], [], []]
for i in range(cropArea[0][0], cropArea[1][0] + 1) :
    for j in range(cropArea[0][1], cropArea[1][1] + 1) :
        t_shadow = int(shadowTime[i, j])
        if t_shadow >= numFrames :
            continue
        p0 = planePoints[t_shadow]
        normal = planeNormals[t_shadow]
        ray = pixel2ray(np.array([i, j], dtype=np.float32), K, distortion).flatten()
        t = np.dot(p0, normal) / np.dot(ray, normal)
        p = t * ray
        if (not (np.isnan(p[0]) or np.isnan(p[1]) or np.isnan(p[2]))) and I_diff[i, j] >= 0.25 :
            reconstructedPoints[0].append(p[0])
            reconstructedPoints[1].append(p[1])
            reconstructedPoints[2].append(p[2])
            reconstructedPoints[3].append(image0[i, j])

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(reconstructedPoints[0], reconstructedPoints[1], reconstructedPoints[2], c=reconstructedPoints[3], cmap='gray', s=0.1)
set_axes_equal(ax)
plt.show()