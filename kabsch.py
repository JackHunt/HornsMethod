import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generateRandomPoints(n, min, max):
	return np.random.uniform(min, max, (n, 3))
	
def applyRandomTransform(points):
	alpha = np.random.uniform(0, 2*np.pi)
	beta = np.random.uniform(0, 2*np.pi)
	gamma = np.random.uniform(0, 2*np.pi)
	Rx = np.matrix([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])
	Ry = np.matrix([[np.cos(beta), -np.sin(beta), 0], [np.sin(beta), np.cos(beta), 0], [0, 0, 1]])
	Rz = np.matrix([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])
	t = np.random.uniform(1, 10, (1, 3))
	
	transPoints = np.dot(points, Rx.transpose())
	transPoints = np.dot(transPoints, Ry.transpose())
	transPoints = np.dot(transPoints, Rz.transpose())
	transPoints += t;
	return (np.asarray(transPoints), Rx, Ry, Rz, t);
	
def applyTransform(R, t, points):
	return np.dot(points, R.transpose()) + t
	
def kabsch(pointsA, pointsB):
	centroidsA = np.mean(pointsA, axis=0)
	centroidsB = np.mean(pointsB, axis=0)
	A = pointsA - centroidsA
	B = pointsB - centroidsB
	C = np.dot(A.transpose(), B)
	U, S, V = np.linalg.svd(C)
	R = np.dot(V.transpose(), U.transpose())
	if(np.linalg.det(R) < 0):
		V[2, :] *= -1
		R = np.dot(V, U.transpose())
	t = np.dot(-R, centroidsA) + centroidsB
	return (R, t)

def displayPoints(pointsA, pointsB):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(pointsA[:, 0], pointsA[:, 1], pointsA[:, 2], c='r')
	ax.scatter(pointsB[:, 0], pointsB[:, 1], pointsB[:, 2], c='b')
	plt.show()

if __name__ == "__main__":
	pointsA = generateRandomPoints(30, -3, 3)
	transformed = applyRandomTransform(pointsA)
	displayPoints(pointsA, transformed[0])
	trans = kabsch(pointsA, transformed[0])
	rectifiedPoints = applyTransform(trans[0], trans[1], pointsA)
	displayPoints(rectifiedPoints, transformed[0])