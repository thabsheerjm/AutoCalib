
import numpy as np
import cv2 
import glob
import utils

class Images:
    def __init__(self,label):
        self.label = label
    
        
        #Properties of the checkerboard
        self.Nx = 9
        self.Ny = 6
        self.corner_world = np.array(([21.5,21.5],[21.5*9,21.5],[21.5*9,21.5*6],[21.5,21.5*6]),dtype= 'float32') # bounds of board in world coords


    def find_corners(self):
        img = cv2.imread(self.label)
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
        ret, corner = cv2.findChessboardCorners(img_gray,(self.Nx,self.Ny),None)
        return ret,corner,img

    def show_corners(self,ret,corner,img):
        Img = cv2.drawChessboardCorners(img,(self.Nx,self.Ny),corner,ret)
        cv2.imshow('res',Img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def V_ij(self,h,i,j):

        Vij =  np.array([h[0, i] * h[0, j], h[0, i] * h[1, j] + h[1, i] * h[0, j], h[1, i] * h[1, j],
                        h[2, i] * h[0, j] + h[0, i] * h[2, j],h[2, i] * h[1, j] + h[1, i] * h[2, j],h[2, i] * h[2, j]]).T
        
        return Vij



#main
def main():
    images = (glob.glob('Calibration_Imgs/*.jpg'))
    num_images = len(images)
    corner_points = []
    V = np.zeros((2*num_images,6))  #2nX6
    i = 0
    homography = np.zeros([3,3])
    for lbl in images:
        img = Images(lbl)
        ret,corners,Img = img.find_corners()
        if ret:
            corner_points.append(corners) # 54 points (9x6)
        else:
            continue
        
        '''Display detected corners'''
        # img.show_corners(ret,corners,Img)

        corner_img = np.array(([corners[0][0]],[corners[8][0]],[corners[53][0]],[corners[45][0]]),dtype = 'float32') # bounds of board in image coords

        H, mask = cv2.findHomography(img.corner_world,corner_img)  
        
        V12 = img.V_ij(H,0,1)
        V11 = img.V_ij(H,0,0)
        V22 = img.V_ij(H,1,1)
        Vi = np.vstack([V12.T,(V11-V22).T])
        V[i:i+2,:] = Vi
        i+=2

        homography  = np.dstack([homography,H])  # stack hoographies of n images along depth axis (channel)
        
    
    homographies = homography[:,:,1:]
    K = utils.Intrinsics(V)
    print("Camera Intrinsic Parameter, Initial Guess: \n",K)

    # using K, find extrinsic parameters
    # The optimized parameters
    K_optim ,k1,k2 = utils.optimize(corner_points,homographies,K)
    print("Camera Intrinsic Parameter, Optimized: \n",K_optim)
    print("Distortion Coefficient Vector: \n", [k1,k2])

    # Calculate mean projection error
    Error, reprojected_pts = utils.Reprojection_error(corner_points,homographies,K_optim,k1,k2)
    print("Mean Reprojection Error: \n",Error)
    

if __name__ == '__main__':
    main()