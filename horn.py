#!/usr/bin/python3

import argparse
import numpy as np
import mayavi.mlab as mlab

def generateRandomPoints(n, min, max):
	return np.random.uniform(min, max, (n, 3))
	
def applyRandomTransform(points, rotSigma, transMax, noiseSD):
	#Generate random Euler Angles(I think.. there must be a better way to do this)
	#Seems to work for this demo however.
	variance = rotSigma*rotSigma
	covariance = np.eye(3) * variance
	alpha, beta, gamma = np.random.multivariate_normal(np.zeros(3), covariance)

	#Build rotation matrices for each principal axis.
	Rx = np.matrix([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])
	Ry = np.matrix([[np.cos(beta), -np.sin(beta), 0], [np.sin(beta), np.cos(beta), 0], [0, 0, 1]])
	Rz = np.matrix([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])
	
	#Get random translation vector.
	t = np.random.uniform(1, 10, (1, 3))
	
	#Apply transform.
	transPoints = np.dot(points, Rx.transpose())
	transPoints = np.dot(transPoints, Ry.transpose())
	transPoints = np.dot(transPoints, Rz.transpose())
	transPoints += t

	if noiseSD is not None:
		noiseVariance = noiseSD*noiseSD
		noiseCovariance = np.eye(3) * noiseVariance
		for point in transPoints:
			eps = np.random.multivariate_normal(np.zeros(3), noiseCovariance)
			point += eps

	return (np.asarray(transPoints), Rx, Ry, Rz, t)
	
def applyTransform(R, t, points):
	return np.dot(points, R.transpose()) + t
	
def horn(pointsA, pointsB):
	centroidsA = np.mean(pointsA, axis=0)
	centroidsB = np.mean(pointsB, axis=0)
	A = pointsA - centroidsA
	B = pointsB - centroidsB
	C = np.dot(A.transpose(), B)
	U, S, V = np.linalg.svd(C)
	R = np.dot(V.transpose(), U.transpose())
	if(np.linalg.det(R) < 0):
		s = eye(3)
		s[2][2] *= -1
		R = np.dot(np.dot(s, V), U.transpose())
	t = np.dot(-R, centroidsA) + centroidsB
	return (R, t)

def displayPoints(pointsA, pointsB, pointsARectified):
	def plotPointPair(figName, A, B):
		fig = mlab.figure(figName, bgcolor=(0, 0, 0), size=(500, 500))
		mlab.points3d(A[:, 0], A[:, 1], A[:, 2], color=(0, 1, 0), scale_factor=0.5, opacity=0.5)
		mlab.points3d(B[:, 0], B[:, 1], B[:, 2], color=(0, 0, 1), scale_factor=0.5, opacity=0.5)
	plotPointPair("BeforeTransform", pointsA, pointsB)
	plotPointPair("AfterTransform", pointsARectified, pointsB)
	mlab.show()

def runDemo(numPoints, bcMin, bcMax, sdRad, transMax, noiseSD):
	pointsA = generateRandomPoints(numPoints, bcMin, bcMax)
	transformed = applyRandomTransform(pointsA, sdRad, transMax, noiseSD)
	trans = horn(pointsA, transformed[0])
	rectifiedPoints = applyTransform(trans[0], trans[1], pointsA)
	displayPoints(pointsA, transformed[0], rectifiedPoints)

if __name__ == "__main__":
	argParser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
	argParser.add_argument('num_points', help="Number of test points to generate.")
	argParser.add_argument('bc_min', help="Min x,y,z value for genrated points(within bounding cube).")
	argParser.add_argument('bc_max', help="Max x,y,z value for genrated points(within bounding cube).")
	argParser.add_argument('--sd_radians', help="Standard Deviation of randomised Euler Angles. [0, 2pi)")
	argParser.add_argument('--trans_max', help="Max value for random translation.")
	argParser.add_argument('--noise_sd', help="Random noise Standard Deviation for transformed points.")
	args = argParser.parse_args()

	sdRad = float(args.sd_radians) if args.sd_radians else 2*np.pi
	transMax = int(args.trans_max) if args.trans_max else 10
	noiseSD = float(args.noise_sd) if args.noise_sd else None

	runDemo(int(args.num_points), int(args.bc_min), int(args.bc_max), sdRad, transMax, noiseSD)