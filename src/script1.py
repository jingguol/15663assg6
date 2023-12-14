import numpy as np
import matplotlib.pyplot as plt
import skimage
import scipy.ndimage
import math
import cv2


verticalTopLeft = (0, 255)
verticalBottomRight = (320, 795)
horizontalTopLeft = (655, 200)
horizontalBottomRight = (767, 835)
shapeVertical = (verticalBottomRight[0] - verticalTopLeft[0] + 1, verticalBottomRight[1] - verticalTopLeft[1] + 1)
shapeHorizontal = (horizontalBottomRight[0] - horizontalTopLeft[0] + 1, horizontalBottomRight[1] - horizontalTopLeft[1] + 1)

startFrame = 1
endFrame = 166
numFrames = endFrame - startFrame + 1

image = plt.imread('../data/frog/000001.jpg')
images = np.zeros((numFrames, image.shape[0], image.shape[1]))

for i in range(startFrame, endFrame + 1) :
    image = plt.imread('../data/frog/' + f'{i:06}' + '.jpg')
    image = np.divide(image, 255.0, dtype=np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2XYZ)
    images[i - startFrame] = image[:, :, 1]

I_max = np.max(images, axis=0)
I_min = np.min(images, axis=0)
I_shadow = (I_max + I_min) * 0.5

I_diff = I_max - I_min
np.savez('intensityDiff.npz', I_diff=I_diff)

for i in range(numFrames) :
    images[i] = images[i] - I_shadow

A_vertical = np.zeros((numFrames, shapeVertical[0], 3))
A_horizontal = np.zeros((numFrames, shapeHorizontal[0], 3))

for i in range(numFrames) :
    for j in range(verticalTopLeft[0], verticalBottomRight[0] + 1) :
        for k in range(verticalTopLeft[1], verticalBottomRight[1] + 1) :
            if images[i, j, k] < 0 and images[i, j, k+1] > 0 :
                A_vertical[i, j-verticalTopLeft[0]] = np.array([j, k, 1.0])
                break
    for j in range(horizontalTopLeft[0], horizontalBottomRight[0] + 1) :
        for k in range(horizontalTopLeft[1], horizontalBottomRight[1] + 1) :
            if images[i, j, k] < 0 and images[i, j, k+1] > 0 :
                A_horizontal[i, j-horizontalTopLeft[0]] = np.array([j, k, 1.0])
                break

edge_vertical = np.zeros((numFrames, 3))
edge_horizontal = np.zeros((numFrames, 3))

## Per-frame shadow edge
for i in range(numFrames) :
    U, S, V = np.linalg.svd(A_vertical[i], full_matrices=True)
    edge_vertical[i] = V[-1]
    U, S, V = np.linalg.svd(A_horizontal[i], full_matrices=True)
    edge_horizontal[i] = V[-1]

np.savez('shadowEdges.npz', edgeVertical=edge_vertical, edgeHorizontal=edge_horizontal)

# image = plt.imread('../data/frog/000121.jpg')
# line_vertical = edge_vertical[120]
# line_horizontal = edge_horizontal[120]
# x1 = [0, image.shape[0]]
# y1 = [line_vertical[2] / (-line_vertical[1]), (line_vertical[0] * x1[1] + line_vertical[2]) / (-line_vertical[1])]
# x2 = [0, image.shape[0]]
# y2 = [line_horizontal[2] / (-line_horizontal[1]), (line_horizontal[0] * x2[1] + line_horizontal[2]) / (-line_horizontal[1])]
# plt.plot(y1, x1, color='green', linewidth=1.5)
# plt.plot(y2, x2, color='blue', linewidth=1.5)
# plt.imshow(image)
# plt.show()

## Per-pixel shadow edge
image = np.zeros(images[0].shape)
images = np.moveaxis(images, 0, -1)
for i in range(images.shape[0]) :
    for j in range(images.shape[1]) :
        for k in range(images.shape[2] - 1) :
            if images[i, j, k] < 0 and images[i, j, k+1] > 0 :
                image[i, j] = k
                break
image = image / np.max(image)
image = np.digitize(image, np.arange(0.0, 1.0, 1.0 / 32))
image = image * (1.0 / 32)
plt.imshow(image, cmap='gray')
plt.show()

np.savez('shadowTime.npz', shadowTime=image)