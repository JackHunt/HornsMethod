#!/usr/bin/python3

import argparse
import numpy as np
import mayavi.mlab as mlab

def generate_random_points(n, min, max):
    return np.random.uniform(min, max, (n, 3))
        
def apply_random_transform(points, range_rad, trans_max, noise_sd):
    #Generate random Euler Angles(I think.. there must be a better way to do this)
    #Seems to work for this demo however.
    alpha, beta, gamma = np.random.uniform(0, range_rad, 3)

    #Build rotation matrices for each principal axis.
    Rx = np.matrix([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])
    Ry = np.matrix([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
    Rz = np.matrix([[1, 0, 0], [0, np.cos(gamma), -np.sin(gamma)], [0, np.sin(gamma), np.cos(gamma)]])
        
    #Get random translation vector.
    t = np.random.uniform(1, 10, (1, 3))
        
    #Apply transform.
    trans_points = np.dot(points, Rx.transpose())
    trans_points = np.dot(trans_points, Ry.transpose())
    trans_points = np.dot(trans_points, Rz.transpose())
    trans_points += t

    if noise_sd is not None:
        noise_variance = noise_sd * noise_sd
        noise_covariance = np.eye(3) * noise_variance
        for point in trans_points:
            eps = np.random.multivariate_normal(np.zeros(3), noise_covariance)
            point += eps

        return (np.asarray(trans_points), Rx, Ry, Rz, t, np.dot(Rx, np.dot(Ry, Rz)))
        
def apply_transform(R, t, points):
    return np.dot(R, points.transpose()).transpose() + t
        
def horn(P, Q):
    if P.shape != Q.shape:
        print("Matrices P and Q must be of the same dimensionality")
        sys.exit(1)

    centroids_P = np.mean(P, axis=1)
    centroids_Q = np.mean(Q, axis=1)
    A = P - np.outer(centroids_P, np.ones(P.shape[1]))
    B = Q - np.outer(centroids_Q, np.ones(Q.shape[1]))
    C = np.dot(A, B.transpose())
    U, S, V = np.linalg.svd(C)
    R = np.dot(V.transpose(), U.transpose())
    if(np.linalg.det(R) < 0):
        L = np.eye(3)
        L[2][2] *= -1

    R = np.dot(V.transpose(), np.dot(L, U.transpose()))
    t = np.dot(-R, centroids_P) + centroids_Q
    return (R, t)

def display_points(points_A, points_B, points_A_rectified):
    def plot_point_pair(fig_name, A, B):
        fig = mlab.figure(fig_name, bgcolor=(0, 0, 0), size=(800, 800))
        mlab.points3d(A[:, 0], A[:, 1], A[:, 2], color=(0, 1, 0), scale_factor=0.5, opacity=0.5)
        mlab.points3d(B[:, 0], B[:, 1], B[:, 2], color=(0, 0, 1), scale_factor=0.5, opacity=0.5)
    plot_point_pair("BeforeTransform", points_A, points_B)
    plot_point_pair("AfterTransform", points_A_rectified, points_B)
    mlab.show()

def run_demo(num_points, bc_min, bc_max, range_rad, trans_max, noise_sd):
    #Generate random points and apply random rigid transform.
    points_A = generate_random_points(num_points, bc_min, bc_max)
    transformed = apply_random_transform(points_A, range_rad, trans_max, noise_sd)

    #Apply Horns Method to align the point clouds.
    trans = horn(points_A.transpose(), transformed[0].transpose())
    rectified_points = apply_transform(trans[0], trans[1], points_A)

    #Give some output.
    display_points(points_A, transformed[0], rectified_points)

    #Calculate RMS.
    diff = transformed[0] - rectified_points
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
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    arg_parser.add_argument('num_points', help="Number of test points to generate.")
    arg_parser.add_argument('bc_min', help="Min x,y,z value for genrated points(within bounding cube).")
    arg_parser.add_argument('bc_max', help="Max x,y,z value for genrated points(within bounding cube).")
    arg_parser.add_argument('--range_radians', help="Range of randomised Euler Angles. [0, 2pi)")
    arg_parser.add_argument('--trans_max', help="Max value for random translation.")
    arg_parser.add_argument('--noise_sd', help="Random noise Standard Deviation for transformed points.")
    args = arg_parser.parse_args()

    range_rad = float(args.range_radians) if args.range_radians else 2*np.pi
    trans_max = int(args.trans_max) if args.trans_max else 10
    noise_sd = float(args.noise_sd) if args.noise_sd else None

    run_demo(int(args.num_points), float(args.bc_min), float(args.bc_max), range_rad, trans_max, noise_sd)
