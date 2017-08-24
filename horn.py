#!/usr/bin/python3

import argparse
import numpy as np
import mayavi.mlab as mlab

def generateRandomPoints(n, min, max):
        return np.random.uniform(min, max, (n, 3))
        
def applyRandomTransform(points, rangeRad, transMax, noiseSD):
        #Generate random Euler Angles(I think.. there must be a better way to do this)
        #Seems to work for this demo however.
        alpha, beta, gamma = np.random.uniform(0, rangeRad, 3)

        #Build rotation matrices for each principal axis.
        Rx = np.matrix([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])
        Ry = np.matrix([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
        Rz = np.matrix([[1, 0, 0], [0, np.cos(gamma), -np.sin(gamma)], [0, np.sin(gamma), np.cos(gamma)]])
        
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

        return (np.asarray(transPoints), Rx, Ry, Rz, t, np.dot(Rx, np.dot(Ry, Rz)))
        
def applyTransform(R, t, points):
        return np.dot(R, points.transpose()).transpose() + t
        
def horn(P, Q):
        if P.shape != Q.shape:
            print("Matrices P and Q must be of the same dimensionality")
            sys.exit(1)

        centroidsP = np.mean(P, axis=1)
        centroidsQ = np.mean(Q, axis=1)
        A = P - np.outer(centroidsP, np.ones(P.shape[1]))
        B = Q - np.outer(centroidsQ, np.ones(Q.shape[1]))
        C = np.dot(A, B.transpose())
        U, S, V = np.linalg.svd(C)
        R = np.dot(V.transpose(), U.transpose())
        if(np.linalg.det(R) < 0):
                L = np.eye(3)
                L[2][2] *= -1
                R = np.dot(V.transpose(), np.dot(L, U.transpose()))
        t = np.dot(-R, centroidsP) + centroidsQ
        return (R, t)

def displayPoints(pointsA, pointsB, pointsARectified):
        def plotPointPair(figName, A, B):
                fig = mlab.figure(figName, bgcolor=(0, 0, 0), size=(800, 800))
                mlab.points3d(A[:, 0], A[:, 1], A[:, 2], color=(0, 1, 0), scale_factor=0.5, opacity=0.5)
                mlab.points3d(B[:, 0], B[:, 1], B[:, 2], color=(0, 0, 1), scale_factor=0.5, opacity=0.5)
        plotPointPair("BeforeTransform", pointsA, pointsB)
        plotPointPair("AfterTransform", pointsARectified, pointsB)
        mlab.show()

def runDemo(numPoints, bcMin, bcMax, rangeRad, transMax, noiseSD):
        #Generate random points and apply random rigid transform.
        pointsA = generateRandomPoints(numPoints, bcMin, bcMax)
        transformed = applyRandomTransform(pointsA, rangeRad, transMax, noiseSD)

        #Apply Horns Method to align the point clouds.
        trans = horn(pointsA.transpose(), transformed[0].transpose())
        rectifiedPoints = applyTransform(trans[0], trans[1], pointsA)

        #Give some output.
        displayPoints(pointsA, transformed[0], rectifiedPoints)

        #Calculate RMS.
        diff = transformed[0] - rectifiedPoints
        rmse = np.sqrt((diff**2).mean())

        #Give output.
        np.set_printoptions(formatter={'float_kind':'{:f}'.format})
        print("Registration RMSE: %f" % rmse)
        print("True rotation:")
        print(transformed[5])
        print("True translation:")
        print(transformed[4])
        print("Estimated rotation:")
        print(trans[0])
        print("Estimated translation:")
        print(trans[1])

if __name__ == "__main__":
        argParser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
        argParser.add_argument('num_points', help="Number of test points to generate.")
        argParser.add_argument('bc_min', help="Min x,y,z value for genrated points(within bounding cube).")
        argParser.add_argument('bc_max', help="Max x,y,z value for genrated points(within bounding cube).")
        argParser.add_argument('--range_radians', help="Range of randomised Euler Angles. [0, 2pi)")
        argParser.add_argument('--trans_max', help="Max value for random translation.")
        argParser.add_argument('--noise_sd', help="Random noise Standard Deviation for transformed points.")
        args = argParser.parse_args()

        rangeRad = float(args.range_radians) if args.range_radians else 2*np.pi
        transMax = int(args.trans_max) if args.trans_max else 10
        noiseSD = float(args.noise_sd) if args.noise_sd else None

        runDemo(int(args.num_points), float(args.bc_min), float(args.bc_max), rangeRad, transMax, noiseSD)
