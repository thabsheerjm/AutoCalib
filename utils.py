
import math
import numpy as np
import scipy as scp



def Intrinsics(V):
    # solve Vb = 0
    u,s,Vmat_T = np.linalg.svd(V)
    b = Vmat_T[:][-1] # get the last row as the solution for b 
    # substitute the values of B11 to B33 to get B matrix and find the alpha, beta, gamma, u0, v0
    alpha = math.sqrt((b[0]*b[2]*b[5] - (b[1]**2)*b[5] - b[0]*(b[4]**2) + 2*b[1]*b[3]*b[4] - b[2]*(b[3]**2))/((b[0]*b[2] - b[1]**2)*b[0]))
    beta = math.sqrt(((b[0]*b[2]*b[5] - (b[1]**2)*b[5] - b[0]*(b[4]**2) + 2*b[1]*b[3]*b[4] - b[2]*(b[3]**2))/((b[0]*b[2] - b[1]**2)**2))*b[0])
    gamma = math.sqrt(( b[0]*b[2]*b[5] - (b[1]**2)*b[5] - b[0]*(b[4]**2) + 2*b[1]*b[3]*b[4] - b[2]*(b[3]**2))/(((b[0]*b[2] - b[1]**2)**2)*b[0])) * b[1] * -1
    u0 = (b[1]*b[4] - b[2]*b[3]) / (b[0]*b[2] - b[1]**2)
    v0 = (b[1]*b[3] - b[0]*b[4]) / (b[0]*b[2] - b[1]**2)

    Kmat = np.array([[alpha, gamma, u0],[0, beta, v0],[0, 0, 1]])
    return Kmat


def calculate_lambda(K_inv,H):
	#Take the avaerage of norm for precision
	return ((np.linalg.norm(np.matmul(K_inv,H[:,0]))+(np.linalg.norm(np.matmul(K_inv,H[:,1]))))/2)



def extrinsic(K, H):
    n = H.shape[2]  # number of images
    Extrinsic = []
    K_inv = np.linalg.inv(K)
    for i in range(n):
        A = np.dot(K_inv, H[:,:,i])
        Lambda = np.linalg.norm(A[:,0])
        A = A / Lambda
        r1 = A[:,0]
        r2 = A[:,1]
        r3 = np.cross(r1, r2)
        t = A[:,2]
        R = np.column_stack((r1, r2, r3))
        extrinsic = np.zeros((3,4))
        extrinsic[:,:-1] = R
        extrinsic[:,-1] = t
        Extrinsic.append(extrinsic)
    return Extrinsic



def optimization_function(camera_params, corner_points, extrinsic_matrices):
    K = np.array([[camera_params[0], camera_params[4], camera_params[2]],
                  [0, camera_params[1], camera_params[3]],
                  [0, 0, 1]])
    k1, k2 = camera_params[5:]
    u0, v0 = camera_params[2:4]
    World_XY = np.array([[21.5*col, 21.5*row, 0, 1]
                         for row in range(1, 7)
                         for col in range(1, 10)])
    error = []
    for corner_points_i, extrinsic_i in zip(corner_points, extrinsic_matrices):
        for corner_point, world_point in zip(corner_points_i, World_XY):
            projected = np.dot(extrinsic_i, world_point)
            projected = projected / projected[-1]
            x, y = projected[:2]
            U = np.dot(K, projected)
            U = U/U[-1]
            u, v = U[:2]
            T = x**2 + y**2
            u_hat = u + (u - u0)*(k1*T + k2*(T**2))
            v_hat = v + (v - v0)*(k1*T + k2*(T**2))
            error.extend([corner_point[0,0] - u_hat, corner_point[0,1] - v_hat])
    return np.array(error).flatten()




def optimize(corner_points, Homographies, K):
    extrinsic_matrices = extrinsic(K, Homographies)
    initial_params = np.array([K[0,0], K[1,1], K[0,2], K[1,2], K[0,1], 0, 0])
    optimized_params = scp.optimize.least_squares(optimization_function, initial_params,
                                                    method='lm', args=[corner_points, extrinsic_matrices])
    alpha, beta, u0, v0, gamma, k1, k2 = optimized_params.x
    new_A = np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]])
    return new_A, k1, k2






def Reprojection_error(corner_points, homographies, K, k1, k2):
    extrinsic_matrices = extrinsic(K, homographies)
    u0, v0 = K[0,2], K[1,2]
    World_XY = np.array([[21.5*col, 21.5*row, 0, 1] for row in range(1, 7) for col in range(1, 10)])
    errors, reprojected_points = [], []
    for cpoints, ext in zip(corner_points, extrinsic_matrices):
        for pts, world in zip(cpoints, World_XY):
            projected = np.dot(ext, world) 
            projected /= projected[-1]
            x, y = projected[:2]
            U = np.dot(K, projected) 
            U /= U[-1]
            u, v = U[:2]
            T = x**2 + y**2
            u_hat = u + (u - u0)*(k1*T + k2*(T**2))
            v_hat = v + (v - v0)*(k1*T + k2*(T**2))
            reprojected_points.append([u_hat, v_hat])
            errors.append(math.sqrt((pts[0,0] - u_hat)**2 + (pts[0,1] - v_hat)**2))
    reprojected_points = np.reshape(reprojected_points, (13, 54, 1, 2))
    return np.mean(errors), reprojected_points
