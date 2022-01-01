                          ##    Comparative study of detectors, descriptors and matching    ##
                          ##                     of points of interest                      ##         
                          ##                Work done by : EL GHEBOULI Ayoub                ## 


## Objective: In this project, we tested and evaluated the performance of several most used interest point detectors 
# such as (SIFT, AKAZE, ORB, BRISK, KAZE, FAST, STAR and MSER), several descriptors such as (SIFT, AKAZE, ORB, BRISK, 
# KAZE, FREAK, LATCH, LUCID and BRIEF) and several matching methods such as (Brute-Force L1, Brute-Force L2 and 
# Brute-Force HAMMING), in order to know the most suitable method for a given scenario.

# ................................................................................
# ................................................................................

## Importation of libraries
import matplotlib.pyplot as plt # For displaying the figures
import cv2 # opencv
import numpy as np # For numerical calculations
import pykitti # to read our database
import time # for the calculation of the execution time
from prettytable import PrettyTable # To view the displayboards on the console

## Reading of our database composed of 50 images for 4 cameras 
basedir = 'KITTI_SAMPLE//RAW'
date = '2011_09_26'
drive = '0009'
data = pykitti.raw(basedir, date, drive, frames=range(0, 50, 1))

# ...................................................................................................................
# ...................................................................................................................
########################### I.1 Data preparation 
# ...................................................................................................................
# ...................................................................................................................

## Scenario 1 (Intensity): Function that returns 8 images with intensity changes from an I image.
def get_cam_intensity_8Img(image0, val_b, val_c): # val_b, val_c must be 2 verctors with 4 values each
    imageO = np.array(image0)
    image = np.array(image0, dtype=np.uint16) # transformation of the image into uint16 so that each pixel of the 
#                                               image will have the same intensity change (min value = 0, max value = 65535)
    I0 = np.zeros((image.shape[0], image.shape[1], image.shape[2])) # creation of empty image of 3 chanels to fill it afterwards
    List8Img = list([I0, I0, I0, I0, I0, I0, I0, I0]) # list of our 8 images that we will create 
    for i in range(len(val_b)): # for I + b, with: b ∈ [-30 : 20 : +30]
        I =  image + val_b[i]
        List8Img[i] = I.astype(int)
        List8Img[i][List8Img[i] > 255] = 255 # set pixels with intensity > 255 to 255 
        List8Img[i][List8Img[i] < 0] = 0 # set the pixels with intensity < 0 to the value of 0
        List8Img[i] = np.array(List8Img[i], dtype=np.uint8) # image transformation to uint8
    for j in range(len(val_c)): # for I ∗ c, with: c ∈ [0.7 : 0.2 : 1.3].
        I =  image * val_c[j]
        List8Img[j+4] = I.astype(int)
        List8Img[j+4][List8Img[j+4] > 255] = 255 # set pixels with intensity > 255 to 255 
        List8Img[j+4][List8Img[j+4] < 0] = 0 # set the pixels with intensity < 0 to the value of 0
        List8Img[j+4] = np.array(List8Img[j+4], dtype=np.uint8) # transform image to uint8 (min value = 0, max value = 255)
    return imageO, List8Img   
# ................................................................................

## Scenario 2 (Scale): Function that takes as input the index of the camera, the index of the image n, and a scale, it returns 
#                      a couple (I, Iscale). In the following, we will work with 7 images with a scale change Is : s ∈]1.1 : 0.2 : 2.3].
def get_cam_scale(camN, n, s): 
    cameras = list([data.get_cam0(n), data.get_cam1(n), data.get_cam2(n), data.get_cam3(n)]) # choose the type of camera and the image index
    Img = cameras[camN]
    Img = np.array(Img) # transform the image into an array type
    ImgScale = cv2.resize(Img, (0, 0), fx=s, fy=s, interpolation = cv2.INTER_NEAREST) # opencv resize function with INTER_NEAREST interpolation
    I_Is = list([Img, ImgScale]) # list of 2 images (original image and scaled image)
    return I_Is 
# ................................................................................

## Scenario 3 (Rotation): Function that takes as input the index of the camera, the index of the image n, and a rotation angle, it returns a 
#                         couple (I, Irot), and the rotation matrix. In the following, we will work with 9 images with a change of scale For 
#                         an image I, we will create 9 images (I10, I20...I90) with change of rotation from 10 to 90 with a step of 10.                               
def get_cam_rot(camN, n, r): 
    cameras = list([data.get_cam0(n), data.get_cam1(n), data.get_cam2(n), data.get_cam3(n)]) # choose the type of camera and the image index
    Img = cameras[camN]
    Img = np.array(Img) # transform the image into an array type
    # divide the height and width by 2 to get the center of the image
    height, width = Img.shape[:2] 
    # obtenir les coordonnées du centre de l'image pour créer la matrice de rotation 2D
    center = (width/2, height/2)
    # get the coordinates of the center of the image to create the 2D rotation matrix
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=r, scale=1)
    # rotate the image using cv2.warpAffine
    rotated_image = cv2.warpAffine(Img, rotate_matrix, dsize=(width, height), flags=cv2.INTER_LINEAR)
    couple_I_Ir = list([Img, rotated_image]) # list of 2 images (original image and image with rotation change)
    return rotate_matrix,couple_I_Ir # it also returns the rotation matrix for further use in the rotation evaluation function
# ................................................................................

## Scenario 4 (Rectified stereo pair): Function that takes as input the index of the image n, it returns a stereo pair of images
def get_cam_stereo_rect(n): 
    ImgStereoG = np.array(data.get_rgb(n)[0]) # left image of the stereo couple
    ImgStereoD = np.array(data.get_rgb(n)[1]) # right image of the stereo couple
    stereo = list([ImgStereoG, ImgStereoD]) # list of 2 images of the stereo couple
    return stereo
# ................................................................................

## Scenario 5: Function that takes as input the index of the image n, it returns 2 consecutive images taken by the same 
#              camera (camera 2) with a forward motion.
def get_cam_img_Sén5(n): 
    ImgCam2_t1 = np.array(data.get_cam2(n)) # image at t1
    ImgCam2_t2 = np.array(data.get_cam2(n+1)) # image at t2
    ImgCam2_t1t2 = list([ImgCam2_t1, ImgCam2_t2]) # list of 2 images at t1 and t2
    return ImgCam2_t1t2
# ................................................................................

## Scenario 6: Function that takes as input the index of the image n, it returns a pair of images (It, I′t+1), with: (It, I′t ) 
#              is a stereo pair taken at time t, and (It+1, I′t+1) a stereo pair taken by the same cameras at time t+1.
def get_cam_Sén6(n): 
    ImgStereo_t1_G = np.array(data.get_rgb(n)[0]) # left image of the stereo couple at t1
    ImgStereo_t2_D = np.array(data.get_rgb(n+1)[1]) # right image of the stereo couple at t2
    ImgCam2_t1Gt2D = list([ImgStereo_t1_G, ImgStereo_t2_D]) # list of 2 images obtained
    return ImgCam2_t1Gt2D
# ................................................................................

## Scenario 7: Function of scenario 6 with the image pair (It, I′t+2).
def get_cam_Sén7(n): 
    ImgStereo_t1_G = np.array(data.get_rgb(n)[0]) # left image of the stereo couple at t1
    ImgStereo_t3_D = np.array(data.get_rgb(n+2)[1]) # right image of the stereo couple at t3
    ImgCam2_t1Gt3D = list([ImgStereo_t1_G, ImgStereo_t3_D]) # list of 2 images obtained
    return ImgCam2_t1Gt3D

# ...................................................................................................................
##### I.2 Scenario evaluation: Function for each scenario that returns the percentage of the match of two lists of correct matched points
# ...................................................................................................................

## Evaluation of scenario 1: Function that takes as input the keypoints, the descriptors (of 2 images) and the type of matching, it returns 
#                            the percentage of correct matched points
def evaluate_scenario_1(KP1, KP2, Dspt1, Dspt2, mise_corresp):
# For this scenario1, the evaluation between two images with change of intensity, we must compare only the coordinates (x,y) of the detected 
# points between the two images.

    # creation of a feature matcher
    bf = cv2.BFMatcher(mise_corresp, crossCheck=True) 
    # match the descriptors of the two images
    matches = bf.match(Dspt1,Dspt2)
    # Sort matches by distance
    matches = sorted(matches, key = lambda x:x.distance)
    
    Prob_P = 0
    Prob_N = 0
    
    # A comparison between the coordinates (x,y) of the detected points between the two images => correct and not correct homologous points 
    for i in range(len(matches)):
        m1 = matches[i].queryIdx
        m2 = matches[i].trainIdx
        # the coordinates (x,y) of the points detected in the image 1
        X1 = int(KP1[m1].pt[0]) 
        Y1 = int(KP1[m1].pt[1])
        # the coordinates (x,y) of the points detected in the image 2
        X2 = int(KP2[m2].pt[0])
        Y2 = int(KP2[m2].pt[1])
        
        # comparison between these coordinates (x,y)
        if (abs(X1 - X2) <=2) and (abs(Y1 - Y2) <=2):   #  Tolerance allowance (∼ 1-2 pixels)
            Prob_P += 1 
        else:
            Prob_N += 1   
    # Calculation of the rate (%) of correctly matched homologous points         
    Prob_True = (Prob_P / (Prob_P + Prob_N))*100
    
    return Prob_True
# ................................................................................

## Evaluation du scénario 2: Function that takes as input the keypoints, the descriptors (of 2 images), 
#                            the type of matching and the scale, it returns the percentage of correct matched points
def evaluate_scenario_2(KP1, KP2, Dspt1, Dspt2, mise_corresp,scale):
# For this scenario2, the evaluation between two images with change of scale, we must compare the coordinates (x,y) 
# of the detected points between the two images (I and I_scale), after multiplying by the scale the coordinates 
# of the detected points in I_scale.
    # creation of a feature matcher
    bf = cv2.BFMatcher(mise_corresp, crossCheck=True) 
    # match the descriptors of the two images
    matches = bf.match(Dspt1,Dspt2)
    # Sort matches by distance
    matches = sorted(matches, key = lambda x:x.distance)
    
    Prob_P = 0
    Prob_N = 0
    
    # A comparison between the coordinates (x,y) of the detected points between the two images => correct and not correct homologous points 
    for i in range(len(matches)):
        m1 = matches[i].queryIdx
        m2 = matches[i].trainIdx
        # the coordinates (x,y) of the points detected in the image 1
        X1 = int(KP1[m1].pt[0])
        Y1 = int(KP1[m1].pt[1])
        # the coordinates (x,y) of the points detected in the image 2
        X2 = int(KP2[m2].pt[0])
        Y2 = int(KP2[m2].pt[1])

        if (abs(X1*scale - X2) <=2) and (abs(Y1*scale - Y2) <=2):   #  Tolerance allowance (∼ 1-2 pixels)
            Prob_P += 1 
        else:
            Prob_N += 1   
    # Calculation of the rate (%) of correctly matched homologous points        
    Prob_True = (Prob_P / (Prob_P + Prob_N))*100
    
    return Prob_True
# ................................................................................

## Evaluation of scenario 3: Function that takes as input the keypoints, the descriptors (of 2 images), 
#                            the type of matching, the degree of rotation and the rotation matrix, it returns 
#                            the percentage of correct matched points
def evaluate_scenario_3(KP1, KP2, Dspt1, Dspt2, mise_corresp,rot, rot_matrix):
# For this scenario3, the evaluation between two images with rotation change, we must compare the coordinates (x,y) 
# of the points detected between the two images (I and I_scale), after multiplying by rot_matrix[:2,:2] the coordinates 
# of the points detected in I_rotation by adding a translation rot_matrix[0,2] for x and rot_matrix[1,2] for y.
   
    # ccreation of a feature matcher
    bf = cv2.BFMatcher(mise_corresp, crossCheck=True)
    # match the descriptors of the two images
    matches = bf.match(Dspt1,Dspt2)
    # Sort matches by distance
    matches = sorted(matches, key = lambda x:x.distance)
    
    Prob_P = 0
    Prob_N = 0
    theta = rot*(np.pi/180) # transformation of the degree of rotation into radian
    # A comparison between the coordinates (x,y) of the detected points between the two images => correct and not correct homologous points 
    for i in range(len(matches)):
        m1 = matches[i].queryIdx
        m2 = matches[i].trainIdx
        # the coordinates (x,y) of the points detected in the image 1
        X1 = int(KP1[m1].pt[0])
        Y1 = int(KP1[m1].pt[1])
        # the coordinates (x,y) of the points detected in the image 2
        X2 = int(KP2[m2].pt[0])
        Y2 = int(KP2[m2].pt[1])

        X12 = X1*np.cos(theta) + Y1*np.sin(theta) + rot_matrix[0,2]
        Y12 = -X1*np.sin(theta) + Y1*np.cos(theta) + rot_matrix[1,2]
    
        if (abs(X12 - X2) <=2) and (abs(Y12 - Y2) <=2):   #  Tolerance allowance (∼ 1-2 pixels)
            Prob_P += 1 
        else:
            Prob_N += 1
    # Calculation of the rate (%) of correctly matched homologous points        
    Prob_True = (Prob_P / (Prob_P + Prob_N))*100
    
    return Prob_True
# ................................................................................

## Evaluation of scenario 4: Function that takes as input the keypoints, the descriptors (of 2 images), 
#                            the type type of matching, returns the percentage of correct matched points
def evaluate_scenario_4(KP1, KP2, Dspt1, Dspt2, mise_corresp):
# For this scenario 4, as for scenario 1 but comparing only the y-intercepts of the points detected in the 2 images
    
    # creation of a feature matcher
    bf = cv2.BFMatcher(mise_corresp, crossCheck=True) 
    # match the descriptors of the two images
    matches = bf.match(Dspt1,Dspt2)
    # Sort matches by distance
    matches = sorted(matches, key = lambda x:x.distance)
    
    Prob_P = 0
    Prob_N = 0
    
    # A comparison between the y-coordinates of the detected points between the two images => correct and not correct homologous points 
    for i in range(len(matches)):
        m1 = matches[i].queryIdx
        m2 = matches[i].trainIdx

        Y1 = int(KP1[m1].pt[1])
        Y2 = int(KP2[m2].pt[1])

        if (abs(Y1 - Y2) <=2):   #  Tolerance allowance (∼ 1-2 pixels)
            Prob_P += 1 
        else:
            Prob_N += 1   
    # Calculation of the rate (%) of correctly matched homologous points        
    Prob_True = (Prob_P / (Prob_P + Prob_N))*100

    return Prob_True, len(matches)
# ................................................................................

## Evaluation of scenarios 5, 6 and 7: Function that takes as input the keypoints, the descriptors (of 2 images), the type type 
#                                      of matching, returns the percentage of correct matched points
def evaluate_scenario_5_6_7(KP1, KP2, Dspt1, Dspt2, mise_corresp):
# For these scenarios, for the evaluation, we used the Fundamental matrix with the method of 8 points and RANSAC, and to 
# calculate the number of homologous Inliers and Outliers

    # creation of a feature matcher
    bf = cv2.BFMatcher(mise_corresp, crossCheck=True) 
    # match the descriptors of the two images
    matches = bf.match(Dspt1,Dspt2)
    # Sort matches by distance
    matches = sorted(matches, key = lambda x:x.distance)

    ## Calculation of the Fundamental matrix with the 8-point method and RANSAC
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = KP1[match.queryIdx].pt
        points2[i, :] = KP2[match.trainIdx].pt
    # Find the matrix Fundamental (h) and the coordinates of the homologous points Inliers and Outliers (mask)
    h, mask = cv2.findFundamentalMat(points1, points2, cv2.RANSAC+cv2.FM_8POINT)

    # Identification of the coordinates of the Inliers homologous points of our two images
    Inliers_Pts1 = points1[mask.ravel()==1]
    Inliers_Pts2 = points2[mask.ravel()==1]
    # Creation of a vect (from zero to the number of points of interest of Inliers)  
    LenPts = len(Inliers_Pts2)
    Vect_in = np.linspace(0, min(Inliers_Pts1.shape[0], Inliers_Pts2.shape[0])-1, LenPts, dtype=np.int) 
    # Convert the coordinates of the points of interest of Inliers for our 2 images to "KeyPoint  
    InkeyPoint1 = [cv2.KeyPoint(x=P[0], y=P[1], _size=1) for P in Inliers_Pts1[Vect_in]] 
    InkeyPoint2 = [cv2.KeyPoint(x=P[0], y=P[1], _size=1) for P in Inliers_Pts2[Vect_in]]
    # Filter the coordinates of the homologous points of Inliers from the set of points
    Inliers_match = [cv2.DMatch(_imgIdx=0, _queryIdx=i, _trainIdx=i,_distance=0) for i in range(len(InkeyPoint2))]

    # Identification of the coordinates of the homologous points Outliers of our two images
    Outliers_Pts1 = points1[mask.ravel()==0]
    Outliers_Pts2 = points2[mask.ravel()==0]
    ## Creation of a vect (from zero to the number of homologous points of Outliers)  
    LenPts = len(Outliers_Pts2)
    Vect_out = np.linspace(0, min(Outliers_Pts1.shape[0], Outliers_Pts2.shape[0])-1, LenPts, dtype=np.int) 
    ## Convert the coordinates of the points of interest of Outliers for our 2 images to "KeyPoint 
    OutkeyPoint1 = [cv2.KeyPoint(x=P[0], y=P[1], _size=1) for P in Outliers_Pts1[Vect_out]] 
    OutkeyPoint2 = [cv2.KeyPoint(x=P[0], y=P[1], _size=1) for P in Outliers_Pts2[Vect_out]]
    ## Filter the coordinates of the homologous points of Outliers from the set of points
    Outliers_match = [cv2.DMatch(_imgIdx=0, _queryIdx=i, _trainIdx=i,_distance=0) for i in range(len(OutkeyPoint2))]

    # Calculation of the rate (%) of correctly matched homologous points 
    Prob_P = np.shape(Inliers_match)[0] # number of Inliers points
    Prob_N = np.shape(Outliers_match)[0] # number of points Outliers    
    Prob_True = (Prob_P / (Prob_P + Prob_N))*100

    return Prob_True, len(matches)

# ...................................................................................................................
# ...................................................................................................................
########################### II. Evaluation of matching methods for scenarios 1, 2 and 3
# ...................................................................................................................
# ...................................................................................................................

# Initialization of our methods of detectors and descriptors (17 methods)
### detectors/descriptors 
sift  = cv2.xfeatures2d.SIFT_create()
akaze = cv2.AKAZE_create()
orb   = cv2.ORB_create()
brisk = cv2.BRISK_create()
kaze  = cv2.KAZE_create()
### detectors
fast  = cv2.FastFeatureDetector_create()
star  = cv2.xfeatures2d.StarDetector_create()
mser  = cv2.MSER_create()
### descriptors 
freak = cv2.xfeatures2d.FREAK_create()
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
lucid = cv2.xfeatures2d.LUCID_create()
latch = cv2.xfeatures2d.LATCH_create()
# lists of the different detectors, descriptors and matching methods
DetectDescript = list([sift, akaze, orb, brisk, kaze])
Detecteurs     = list([fast, star, mser])
Descripteurs   = list([freak, brief, lucid, latch])
mise_en_correspondances2 = list([cv2.NORM_L1, cv2.NORM_L2])
mise_en_correspondances3 = list([cv2.NORM_L1, cv2.NORM_L2, cv2.NORM_HAMMING])
# ................................................................................

################ Sénario 1 
Img0 = data.get_cam2(0) # Original image
Img0 = np.array(Img0)
val_b = np.array([-30, -10, 10, 30]) # b ∈ [−30 : 20 : +30] 
val_c = np.array([0.7, 0.9, 1.1, 1.3]) # c ∈ [0.7 : 0.2 : 1.3].
nbre_img = len(val_b) + len(val_c) # number of intensity change values ==> number of test images

## 2 matrices of the rates of scenario 1, the first one gathers the rates for each image, each non-binary method 
# (same detectors and descriptors), and each type of matching (without bf.HAMMING). And the other one groups the 
# rates for each image, each method binary method (different detectors and descriptors), and each type of matching (with bf.HAMMING). 
Taux_intensity1 = np.zeros((nbre_img, len(mise_en_correspondances2), len(DetectDescript)))
Taux_intensity2 = np.zeros((nbre_img, len(mise_en_correspondances3), len(Detecteurs), len(Descripteurs)))

img1, HuitImg1 = get_cam_intensity_8Img(Img0, val_b, val_c) # use the intensity change images (I+b and I*c) 
# for loop to compute rates (%) for intensity change images, matches, binary and non-binary methods 
for k in range(nbre_img): # for the 8 intensity images
    img2 = HuitImg1[k] # image with intensity change
    for c2 in range(len(mise_en_correspondances2)): # for bf.L1 and bf.L2 mapping (bf.HAMMING does not work for most non-binary methods)
        mise_en_corresp = mise_en_correspondances2[c2] 
        for ii in range(len(DetectDescript)):
            methode = DetectDescript[ii] # choose a method from the "DetectDescript" list
            keypoints11, descriptors11 = methode.detectAndCompute(img1, None) # the keypoints and descriptors of the image 1 obtained by the method X
            keypoints22, descriptors22 = methode.detectAndCompute(img2, None) # the keypoints and descriptors of the image 2 obtained by the method X
            # Calculation of the rate (%) of correctly matched homologous points by the X method using the evaluation function of scenario 1
            Taux_intensity1[k, c2, ii] = evaluate_scenario_1(keypoints11, keypoints22, descriptors11, descriptors22, mise_en_corresp)  
    for c3 in range(len(mise_en_correspondances3)): # for bf.L1, bf.L2 and bf.HAMMING mapping
        mise_en_corresp = mise_en_correspondances3[c3]
        for i in range(len(Detecteurs)):    
            methode_keyPt = Detecteurs[i] # choose a detector from the "Detectors" list
            for j in range(len(Descripteurs)):
                methode_dscrpt = Descripteurs[j] # choose a descriptor from the "Descriptors" list
                keypoints1   = methode_keyPt.detect(img1,None) 
                keypoints2   = methode_keyPt.detect(img2,None)
                keypoints1   = methode_dscrpt.compute(img1, keypoints1)[0] # the keypoints of image 1 obtained by the method Y
                keypoints2   = methode_dscrpt.compute(img2, keypoints2)[0] # the keypoints of image 2 obtained by the method Y
                descriptors1 = methode_dscrpt.compute(img1, keypoints1)[1] # the descriptors of the image 1 obtained by the method Y
                descriptors2 = methode_dscrpt.compute(img2, keypoints2)[1] # the descriptors of the image 2 obtained by the method Y  
                # Calculation of the rate (%) of correctly matched homologous points by the Y method using the evaluation function of scenario 1
                Taux_intensity2[k, c3, i, j] = evaluate_scenario_1(keypoints1, keypoints2, descriptors1, descriptors2, mise_en_corresp)
# ................................................................................

################ Scenario 2: Scale
cameraN = 2 # camera index
ImageN = 0 # image index
scale = [1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3] # 7 values of the scale change s ∈]1.1 : 0.2 : 2.3].

## 2 matrices of the rates of scenario 2, the first one groups the rates for each image, each non-binary method (same detectors and descriptors), 
# and each type of matching (without bf.HAMMING). And the other one groups the rates for each image, each binary method (different detectors and 
# descriptors), and each type of matching (with bf.HAMMING). 
Taux_scale1 = np.zeros((len(scale), len(mise_en_correspondances2), len(DetectDescript)))
Taux_scale2 = np.zeros((len(scale), len(mise_en_correspondances3), len(Detecteurs), len(Descripteurs)))
# for loop to calculate rates (%) for scaling images, matching, binary and non-binary methods
for s in range(len(scale)): # for the 7 scale images
    # use the original image and the scaling image (I and Is) 
    img1 = get_cam_scale(cameraN, ImageN, scale[s])[0] # image I
    img2 = get_cam_scale(cameraN, ImageN, scale[s])[1] # image Is
    for c2 in range(len(mise_en_correspondances2)): # for bf.L1 and bf.L2 mapping (bf.HAMMING does not work for most non-binary methods)
        mise_en_corresp = mise_en_correspondances2[c2]
        for ii in range(len(DetectDescript)):
            methode = DetectDescript[ii] # choose a method from the "DetectDescript" list
            keypoints11, descriptors11 = methode.detectAndCompute(img1, None)# the keypoints and descriptors of image 1 obtained by the method X
            keypoints22, descriptors22 = methode.detectAndCompute(img2, None)# the keypoints and descriptors of image 2 obtained by the method X
            # Calculation of the rate (%) of correctly matched homologous points by the X method using the evaluation function of scenario 2
            Taux_scale1[s, c2, ii] = evaluate_scenario_2(keypoints11, keypoints22, descriptors11, descriptors22, mise_en_corresp, scale[s])
    for c3 in range(len(mise_en_correspondances3)): # for bf.L1, bf.L2 and bf.HAMMING mapping
        mise_en_corresp = mise_en_correspondances3[c3]
        for i in range(len(Detecteurs)):    
            methode_keyPt = Detecteurs[i] # choose a detector from the "Detectors" list
            for j in range(len(Descripteurs)):
                methode_dscrpt = Descripteurs[j] # choose a descriptor from the "Descriptors" list  
                keypoints1   = methode_keyPt.detect(img1,None)
                keypoints2   = methode_keyPt.detect(img2,None)
                keypoints1   = methode_dscrpt.compute(img1, keypoints1)[0] # the keypoints of image 1 obtained by the method Y 
                keypoints2   = methode_dscrpt.compute(img2, keypoints2)[0] # the keypoints of image 2 obtained by the method Y 
                descriptors1 = methode_dscrpt.compute(img1, keypoints1)[1] # the descriptors of the image 1 obtained by the method Y 
                descriptors2 = methode_dscrpt.compute(img2, keypoints2)[1] # the descriptors of the image 2 obtained by the method Y           
                # Calculation of the rate (%) of correctly matched homologous points by the Y method using the evaluation function of scenario 2
                Taux_scale2[s, c3, i, j] = evaluate_scenario_2(keypoints1, keypoints2, descriptors1, descriptors2, mise_en_corresp, scale[s])
# ................................................................................

################ Scenario 3: Rotation
cameraN = 2 # camera index
ImageN = 0 # image index
rot = [10, 20, 30, 40, 50, 60, 70, 80, 90] # 9 values of rotation change, rotations from 10 to 90 with a step of 10.

## 2 matrices of the rates of scenario 3, the first one groups the rates for each image, each non-binary method (same detectors and descriptors), 
# and each type of matching (without bf.HAMMING). And the other one groups the rates for each image, each binary method (different detectors and 
# descriptors), and each type of matching (with bf.HAMMING). 
Taux_rot1 = np.zeros((len(rot), len(mise_en_correspondances2), len(DetectDescript)))
Taux_rot2 = np.zeros((len(rot), len(mise_en_correspondances3), len(Detecteurs), len(Descripteurs)))
# for loop to compute rates (%) for rotation change images, matches, binary and non-binary methods
for r in range(len(rot)):
    # use the rotation matrix, the original image and the rotation change matrix (I and Ir) 
    rot_matrix, img = get_cam_rot(cameraN, ImageN, rot[r])
    img1 = img[0] # image I
    img2 = img[1] # image Ir
    for c2 in range(len(mise_en_correspondances2)): # for bf.L1 and bf.L2 mappings (bf.HAMMING does not work for most non-binary methods)
        mise_en_corresp = mise_en_correspondances2[c2]
        for ii in range(len(DetectDescript)):
            methode = DetectDescript[ii] # choose a method from the "DetectDescript" list
            keypoints11, descriptors11 = methode.detectAndCompute(img1, None)# the keypoints and descriptors of image 1 obtained by the method X
            keypoints22, descriptors22 = methode.detectAndCompute(img2, None)# the keypoints and descriptors of image 2 obtained by the method X
            # Calculation of the rate (%) of correctly matched homologous points by the X method using the evaluation function of scenario 3
            Taux_rot1[r, c2, ii] = evaluate_scenario_3(keypoints11, keypoints22, descriptors11, descriptors22, mise_en_corresp, rot[r], rot_matrix)  
    for c3 in range(len(mise_en_correspondances3)): # for bf.L1, bf.L2 and bf.HAMMING mapping
        mise_en_corresp = mise_en_correspondances3[c3] 
        for i in range(len(Detecteurs)):    
            methode_keyPt = Detecteurs[i] # choose a detector from the "Detectors" list
            for j in range(len(Descripteurs)):
                methode_dscrpt = Descripteurs[j] # choose a descriptor from the "Descriptors" list    
                keypoints1   = methode_keyPt.detect(img1,None)
                keypoints2   = methode_keyPt.detect(img2,None)
                keypoints1   = methode_dscrpt.compute(img1, keypoints1)[0]# the keypoints of image 1 obtained by the method Y
                keypoints2   = methode_dscrpt.compute(img2, keypoints2)[0]# the keypoints of image 2 obtained by the method Y
                descriptors1 = methode_dscrpt.compute(img1, keypoints1)[1]# the descriptors of the image 1 obtained by the method Y
                descriptors2 = methode_dscrpt.compute(img2, keypoints2)[1]# the descriptors of the image 2 obtained by the method Y
                # Calculation of the rate (%) of correctly matched homologous points by the Y method using the evaluation function of scenario 3
                Taux_rot2[r, c3, i, j] = evaluate_scenario_3(keypoints1, keypoints2, descriptors1, descriptors2, mise_en_corresp, rot[r], rot_matrix)

# ..........................................................................................................................
###############  Visualization of the results by 4 figures, one per scenario (two for the 1st) for scenarios 1, 2 and 3
# ..........................................................................................................................

# Binary and non-binary methods used to set the legend
DetectDescript = ['sift', 'akaze', 'orb', 'brisk', 'kaze']
Detecteurs     = ['fast-', 'star-', 'mser-']
Descripteurs   = ['freak', 'brief', 'lucid', 'latch']

c2 = 1 # for non-binary methods "DetectDescript" (c2=0 for bf.L1, c2=1 for bf.L2)
c3 = 2 # for binary methods "Detectors with Descriptors" (c2=0 for bf.L1, c2=1 for bf.L2, c2=2 for bf.HAMMING)
# To choose the type of mapping for our binary and non-binary methods (this is for a good visualization, to 
# avoid plotting 46 curves and plotting only 17 curves in each figure)

# Number of colors to use for all curves
NUM_COLORS = len(DetectDescript) + (len(Detecteurs)*len(Descripteurs)) # NUM_COLORS = 17

LINE_STYLES = ['solid', 'dashed', 'dotted'] # style of the curve
NUM_STYLES = len(LINE_STYLES)
cm = plt.get_cmap('gist_rainbow')
num = -1 # for plot
# Initialization of the 4 figures
fig1 = plt.figure(1,figsize= (13,5))
fig2 = plt.figure(2,figsize= (13,5))
fig3 = plt.figure(3,figsize= (13,5))
fig4 = plt.figure(4,figsize= (13,5))
ax1 = fig1.add_subplot(111)
ax2 = fig2.add_subplot(111)
ax3 = fig3.add_subplot(111)
ax4 = fig4.add_subplot(111)

# for the plot, I have inserted the following link: https://stackoverflow.com/questions/8389636/creating-over-20-unique-legend-colors-using-matplotlib
# for loop to display the results of non-binary methods
for k in range(len(DetectDescript)):
    Taux1_I1 = Taux_intensity1[:4, c2, k]
    Taux1_I2 = Taux_intensity1[4:, c2, k]
    Taux1_S = Taux_scale1[:, c2, k]
    Taux1_R = Taux_rot1[:, c2, k]

    lines_I1 = ax1.plot(val_b, Taux1_I1, linewidth=2, label = DetectDescript[k]) # for the figure of the intensity change results (I+b)
    lines_I2 = ax2.plot(val_c, Taux1_I2, linewidth=2, label = DetectDescript[k]) # for the figure of intensity change results (I*c)
    lines_S = ax3.plot(scale, Taux1_S, linewidth=2, label = DetectDescript[k]) # for the scaling results figure 
    lines_R = ax4.plot(rot, Taux1_R, linewidth=2, label = DetectDescript[k]) # for the figure of the results of rotation change

    num += 1 # to take each time the loop turns a different color and curve style
    # for the color and style of the curve for the results of the 3 scenarios
    lines_I1[0].set_color(cm(num//NUM_STYLES*float(NUM_STYLES)/NUM_COLORS))
    lines_I1[0].set_linestyle(LINE_STYLES[num%NUM_STYLES])
    lines_I2[0].set_color(cm(num//NUM_STYLES*float(NUM_STYLES)/NUM_COLORS))
    lines_I2[0].set_linestyle(LINE_STYLES[num%NUM_STYLES])
    lines_S[0].set_color(cm(num//NUM_STYLES*float(NUM_STYLES)/NUM_COLORS))
    lines_S[0].set_linestyle(LINE_STYLES[num%NUM_STYLES])
    lines_R[0].set_color(cm(num//NUM_STYLES*float(NUM_STYLES)/NUM_COLORS))
    lines_R[0].set_linestyle(LINE_STYLES[num%NUM_STYLES])

# for loop to display the results of binary methods
for i in range(len(Detecteurs)):
    for j in range(len(Descripteurs)):
        Taux2_I1 = Taux_intensity2[:4,c3,i,j]
        Taux2_I2 = Taux_intensity2[4:,c3,i,j]
        Taux2_S = Taux_scale2[:,c3,i,j]
        Taux2_R = Taux_rot2[:,c3,i,j]
       
        lines_I1 = ax1.plot(val_b, Taux2_I1, linewidth=2, label = Detecteurs[i] + Descripteurs[j]) # for the figure of intensity change results (I+b)
        lines_I2 = ax2.plot(val_c, Taux2_I2, linewidth=2, label = Detecteurs[i] + Descripteurs[j]) # for the figure of intensity change results (I*c)
        lines_S = ax3.plot(scale, Taux2_S, linewidth=2, label = Detecteurs[i] + Descripteurs[j]) # for the figure of the results of scale change
        lines_R = ax4.plot(rot, Taux2_R, linewidth=2, label = Detecteurs[i] + Descripteurs[j]) # for the figure of the results of rotation change

        num += 1 # to take each time the loop turns a different style of curve
        # for the color and style of curve for the results of the 3 scenarios
        lines_I1[0].set_color(cm(num//NUM_STYLES*float(NUM_STYLES)/NUM_COLORS))
        lines_I1[0].set_linestyle(LINE_STYLES[num%NUM_STYLES])
        lines_I2[0].set_color(cm(num//NUM_STYLES*float(NUM_STYLES)/NUM_COLORS))
        lines_I2[0].set_linestyle(LINE_STYLES[num%NUM_STYLES])
        lines_S[0].set_color(cm(num//NUM_STYLES*float(NUM_STYLES)/NUM_COLORS))
        lines_S[0].set_linestyle(LINE_STYLES[num%NUM_STYLES])
        lines_R[0].set_color(cm(num//NUM_STYLES*float(NUM_STYLES)/NUM_COLORS))
        lines_R[0].set_linestyle(LINE_STYLES[num%NUM_STYLES])

# The titles of the figures according to the correspondences
if c2 == 0 and c3 == 0:
    ax1.set_title('Results of scenario 1, with bf.L1 for non-binary methods and bf.L1 for binary methods', fontsize=13)
    ax2.set_title('Results of scenario 1, with bf.L1 for non-binary methods and bf.L1 for binary methods', fontsize=13)
    ax3.set_title('Results of scenario 3, with bf.L1 for non-binary methods and bf.L1 for binary methods', fontsize=13)
    ax4.set_title('Results of scenario 4, with bf.L1 for non-binary methods and bf.L1 for binary methods', fontsize=13)
elif c2 == 1 and c3 == 1:
    ax1.set_title('Results of scenario 1, with bf.L2 for non-binary methods and bf.L2 for binary methods', fontsize=13)
    ax2.set_title('Results of scenario 1, with bf.L2 for non-binary methods and bf.L2 for binary methods', fontsize=13)
    ax3.set_title('Results of scenario 3, with bf.L2 for non-binary methods and bf.L2 for binary methods', fontsize=13)
    ax4.set_title('Results of scenario 4, with bf.L2 for non-binary methods and bf.L2 for binary methods', fontsize=13)
elif c2 == 1 and c3 == 2:
    ax1.set_title('Results of scenario 1, with bf.L2 for non-binary methods and bf.HAMMING for binary methods', fontsize=13)
    ax2.set_title('Results of scenario 1, with bf.L2 for non-binary methods and bf.HAMMING for binary methods', fontsize=13)
    ax3.set_title('Results of scenario 3, with bf.L2 for non-binary methods and bf.HAMMING for binary methods', fontsize=13)
    ax4.set_title('Results of scenario 4, with bf.L2 for non-binary methods and bf.HAMMING for binary methods', fontsize=13)

ax1.set_xlabel('Intensity changing (Img +/- value)', fontsize=12) # x-axis title of the figure
ax1.set_ylabel('Correctly matched point rates %', fontsize=12) # title of y-axis of the figure
ax1.legend(loc= 'center left', bbox_to_anchor=(1, 0.5), fontsize= 10, handlelength = 2) # legend :(loc=2 <=> Location String = 'upper left')

# ax2.set_title('Taux de points correctement appariés pour différentes méthodes d'appariement en fonction du changement d'intensité', fontsize=13)
ax2.set_xlabel('Intensity changing (Img * value)', fontsize=12) # titre de l'axe x de la figure
ax2.set_ylabel('Correctly matched point rates %', fontsize=12) # titre d'axe y de la figure
ax2.legend(loc= 'center left', bbox_to_anchor=(1, 0.5), fontsize= 10, handlelength = 2) # (loc=2 <=> Location String = 'upper left')

# ax3.set_title('Taux de points correctement appariés pour différentes méthodes d'appariement en fonction du changement d'échelle', fontsize=13)
ax3.set_xlabel('Scale changing', fontsize=12) # titre de l'axe x de la figure
ax3.set_ylabel('Correctly matched point rates %', fontsize=12) # titre d'axe y de la figure
ax3.legend(loc= 'center left', bbox_to_anchor=(1, 0.5), fontsize= 10, handlelength = 2) # (loc=2 <=> Location String = 'upper left')

# ax4.set_title('Taux de points correctement appariés pour différentes méthodes d'appariement en fonction du changement de rotation', fontsize=13)
ax4.set_xlabel('Rotation changing', fontsize=12) # titre de l'axe x de la figure
ax4.set_ylabel('Correctly matched point rates %', fontsize=12) # titre d'axe y de la figure
ax4.legend(loc= 'center left', bbox_to_anchor=(1, 0.5), fontsize= 10, handlelength = 2) # (loc=2 <=> Location String = 'upper left')

# Recording and display of the obtained figures
fig1.savefig('Intensity1_changing.png')
fig2.savefig('Intensity1_changing.png')
fig3.savefig('Scale_changing.png')
fig4.savefig('Rotation_changing.png')
plt.show()

# ...................................................................................................................
# ...................................................................................................................
########################### III. Evaluation of matching methods for scenarios 4, 5, 6 and 7 
# ...................................................................................................................
# ...................................................................................................................

################ Scenario 4 
N_S4 = 1 # Set of test images for scenario 4
# Initialization of the results matrices 
Nbre_matches1_S4 = np.zeros((N_S4, len(mise_en_correspondances2), len(DetectDescript))) # for the number of homologous points detected with non-binary methods
Nbre_matches2_S4 = np.zeros((N_S4, len(mise_en_correspondances3), len(Detecteurs), len(Descripteurs))) # for the number of homologous points detected with binary methods
Taux_stereo_rect1_S4 = np.zeros((N_S4, len(mise_en_correspondances2), len(DetectDescript))) # for the number of detected homologous points correctly matched with non-binary methods
Taux_stereo_rect2_S4 = np.zeros((N_S4, len(mise_en_correspondances3), len(Detecteurs), len(Descripteurs))) # for the number of detected homologous points correctly matched with the binary methods
temps_execution1_S4 = np.zeros((N_S4, len(mise_en_correspondances2), len(DetectDescript))) # for the execution time of the non-binary methods
temps_execution2_S4 = np.zeros((N_S4, len(mise_en_correspondances3), len(Detecteurs), len(Descripteurs))) # for the execution time of the binary methods

# for loop to calculate rates (%) for scenario 4 images, matches, binary and non-binary methods
for n in range(N_S4):
    img1 = get_cam_stereo_rect(n)[0] # image 1 left of the stereo couple
    img2 = get_cam_stereo_rect(n)[1] # image 1 right of the stereo pair
    for c2 in range(len(mise_en_correspondances2)): # for bf.L1 and bf.L2 matching (bf.HAMMING does not work for most non-binary methods)
        mise_en_corresp2 = mise_en_correspondances2[c2]
        for ii in range(len(DetectDescript)):
            methode = DetectDescript[ii] # choose a method from the "DetectDescript" list 
            start1 = time.time() # start of execution time for method X
            keypoints11, descriptors11 = methode.detectAndCompute(img1, None)# the keypoints and descriptors of image 1 obtained by method X
            keypoints22, descriptors22 = methode.detectAndCompute(img2, None)# the keypoints and descriptors of image 2 obtained by method X
            # Calculation of the rate (%) of correctly matched homologous points and the number of homologous points by the X method using the evaluation function of scenario 4
            Taux_stereo_rect1_S4[n, c2, ii], Nbre_matches1_S4[n, c2, ii] = evaluate_scenario_4(keypoints11, keypoints22, descriptors11, descriptors22, mise_en_corresp2)  
            end1 = time.time() # end of execution time for method X
            # Computation of the execution_time (in seconds) by the X method using the evaluation function of scenario 4
            temps_execution1_S4[n, c2, ii] = (end1 - start1)
    for c3 in range(len(mise_en_correspondances3)): # for the mappings bf.L1, bf.L2 and bf.HAMMING
        mise_en_corresp3 = mise_en_correspondances3[c3]
        for i in range(len(Detecteurs)):    
            methode_keyPt = Detecteurs[i] # choose a detector from the "Detectors" list
            for j in range(len(Descripteurs)):
                methode_dscrpt = Descripteurs[j] # choose a descriptor from the "Descriptors" list 
                start2 = time.time() # start of the execution time for the method X
                keypoints1   = methode_keyPt.detect(img1,None)
                keypoints2   = methode_keyPt.detect(img2,None)
                keypoints1   = methode_dscrpt.compute(img1, keypoints1)[0]# the image 1 keypoints obtained by the Y method
                keypoints2   = methode_dscrpt.compute(img2, keypoints2)[0]# the keypoints of image 2 obtained by the Y method
                descriptors1 = methode_dscrpt.compute(img1, keypoints1)[1]# the descriptors of image 1 obtained by the Y method
                descriptors2 = methode_dscrpt.compute(img2, keypoints2)[1]# the descriptors of image 2 obtained by the Y method           
                 # Calculation of the rate (%) of correctly matched homologous points and the number of homologous points by the Y method using the evaluation function of scenario 4
                Taux_stereo_rect2_S4[n, c3, i, j], Nbre_matches2_S4[n, c3, i, j] = evaluate_scenario_4(keypoints1, keypoints2, descriptors1, descriptors2, mise_en_corresp3)
                end2 = time.time() # end of execution time for the Y method
                # Compute the execution_time (in seconds) by the Y method using the evaluation function of scenario 4
                temps_execution2_S4[n, c3, i, j] = (end2 - start2) 
# ..........................................................................................................................
###############  Visualization of the results of scenario 4 by table (using the prettytable library)
# ..........................................................................................................................

# La moyenne des resultats sur l'ensemble des images (50 images)
Nbre_matches1m_S4 = np.mean(Nbre_matches1_S4,0)
Nbre_matches2m_S4 = np.mean(Nbre_matches2_S4,0)
Taux_stereo_rect1m_S4 = np.mean(Taux_stereo_rect1_S4,0)
Taux_stereo_rect2m_S4 = np.mean(Taux_stereo_rect2_S4,0)
temps_execution1m_S4 = np.mean(temps_execution1_S4,0)
temps_execution2m_S4 = np.mean(temps_execution2_S4,0)

# concatination des resultats dans un vecteur pour la simplification d'affichage dans le tableau
Nbre_matches_Sén4 = np.concatenate((Nbre_matches1m_S4[0], Nbre_matches2m_S4[0][0], Nbre_matches2m_S4[0][1], Nbre_matches2m_S4[0][2], 
                                  Nbre_matches1m_S4[1], Nbre_matches2m_S4[1][0], Nbre_matches2m_S4[1][1], Nbre_matches2m_S4[1][2], 
                                  Nbre_matches2m_S4[2][0], Nbre_matches2m_S4[2][1], Nbre_matches2m_S4[2][2]), axis=0)
Taux_stereo_rect_Sén4 = np.concatenate((Taux_stereo_rect1m_S4[0], Taux_stereo_rect2m_S4[0][0], Taux_stereo_rect2m_S4[0][1], Taux_stereo_rect2m_S4[0][2], 
                                  Taux_stereo_rect1m_S4[1], Taux_stereo_rect2m_S4[1][0], Taux_stereo_rect2m_S4[1][1], Taux_stereo_rect2m_S4[1][2], 
                                  Taux_stereo_rect2m_S4[2][0], Taux_stereo_rect2m_S4[2][1], Taux_stereo_rect2m_S4[2][2]), axis=0)
temps_execution_Sén4 = np.concatenate((temps_execution1m_S4[0], temps_execution2m_S4[0][0], temps_execution2m_S4[0][1], temps_execution2m_S4[0][2], 
                                  temps_execution1m_S4[1], temps_execution2m_S4[1][0], temps_execution2m_S4[1][1], temps_execution2m_S4[1][2], 
                                  temps_execution2m_S4[2][0], temps_execution2m_S4[2][1], temps_execution2m_S4[2][2]), axis=0)
# The set of methods used to put them in legend
DetectDescript = list(['sift_bf.L1', 'akaze_bf.L1', 'orb_bf.L1', 'brisk_bf.L1', 'kaze_bf.L1', 'fast-freak_bf.L1', 'fast-brief_bf.L1', 
                  'fast-lucid_bf.L1', 'fast-latch_bf.L1', 'star-freak_bf.L1', 'star-brief_bf.L1', 'star-lucid_bf.L1', 'star-latch_bf.L1', 
                  'mser-freak_bf.L1', 'mser-brief_bf.L1', 'mser-lucid_bf.L1', 'mser-latch_bf.L1', 'sift_bf.L2', 'akaze_bf.L2', 'orb_bf.L2', 
                  'brisk_bf.L2', 'kaze_bf.L2', 'fast-freak_bf.L2', 'fast-brief_bf.L2', 'fast-lucid_bf.L2', 'fast-latch_bf.L2', 'star-freak_bf.L2', 
                  'star-brief_bf.L2', 'star-lucid_bf.L2', 'star-latch_bf.L2', 'mser-freak_bf.L2', 'mser-brief_bf.L2', 'mser-lucid_bf.L2', 
                  'mser-latch_bf.L2', 'fast-freak_bf.HAMG', 'fast-brief_bf.HAMG', 'fast-lucid_bf.HAMG', 'fast-latch_bf.HAMG', 'star-freak_bf.HAMG', 
                  'star-brief_bf.HAMG', 'star-lucid_bf.HAMG', 'star-latch_bf.HAMG', 'mser-freak_bf.HAMG', 'mser-brief_bf.HAMG', 'mser-lucid_bf.HAMG', 'mser-latch_bf.HAMG'])

# Initialize the table, then give it a title, and add the column titles for our different results 
x_Sén4 = PrettyTable()
x_Sén4.title = "Résultats du scénario 4"
x_Sén4.field_names = ['Méthode', "Temps d’exécution (s)", "Nombre de points/image", "Taux d’appariement correcte"]

# Organization of the rate of correct matches in descending order for good visualization of the results
Taux_sorted_Sén4 = sorted(Taux_stereo_rect_Sén4, reverse = True)

# Search for the coordinates of this organization to organize the results of the execution times and the number of homologous points detected in their vectors
coords_Sén4 = []
for i in range(len(Taux_sorted_Sén4)):
    coord = np.where(Taux_stereo_rect_Sén4 == Taux_sorted_Sén4[i])[0][0]
    coords_Sén4.append(coord)

# Display the results in the table in descending order of the rate of correct matches
for i in coords_Sén4:
    x_Sén4.add_row([DetectDescript[i], round(temps_execution_Sén4[i], 3), round(Nbre_matches_Sén4[i], 3), round(Taux_stereo_rect_Sén4[i], 3)])
print(x_Sén4)
#.........................................................................................

################ Scenario 5:
N_S5 = 49 # Set of test images for scenario 5
# Initialization of the results matrices
Nbre_matches1_S5 = np.zeros((N_S5, len(mise_en_correspondances2), len(DetectDescript)))
Nbre_matches2_S5 = np.zeros((N_S5, len(mise_en_correspondances3), len(Detecteurs), len(Descripteurs)))
Taux_stereo_rect1_S5 = np.zeros((N_S5, len(mise_en_correspondances2), len(DetectDescript)))
Taux_stereo_rect2_S5 = np.zeros((N_S5, len(mise_en_correspondances3), len(Detecteurs), len(Descripteurs)))
temps_execution1_S5 = np.zeros((N_S5, len(mise_en_correspondances2), len(DetectDescript)))
temps_execution2_S5 = np.zeros((N_S5, len(mise_en_correspondances3), len(Detecteurs), len(Descripteurs)))

# for loop to compute rates (%) for scenario 5 images, matching, binary and non-binary methods
for n in range(N_S5):
    img1 = get_cam_img_Sén5(n)[0] # image 1 of a camera at t1
    img2 = get_cam_img_Sén5(n)[1] # image 2 of the same camera at t2
    for c2 in range(len(mise_en_correspondances2)):# for bf.L1 and bf.L2 matches (bf.HAMMING does not work for most non-binary methods)
        mise_en_corresp2 = mise_en_correspondances2[c2]
        for ii in range(len(DetectDescript)):
            methode = DetectDescript[ii]# choose a method from the "DetectDescript" list 
            start1 = time.time() # start execution time for method X 
            keypoints11, descriptors11 = methode.detectAndCompute(img1, None)# the keypoints and descriptors of image 1 obtained by method X
            keypoints22, descriptors22 = methode.detectAndCompute(img2, None)# the keypoints and descriptors of image 2 obtained by method X
            # Calculation of the rate (%) of correctly matched homologous points and the number of homologous points by the X method using the evaluation function of scenario 5
            Taux_stereo_rect1_S5[n, c2, ii], Nbre_matches1_S5[n, c2, ii] = evaluate_scenario_5_6_7(keypoints11, keypoints22, descriptors11, descriptors22, mise_en_corresp2)  
            end1 = time.time() # end of execution time for method X
            # Calculation of the execution time (in seconds) by the X method using the evaluation function of scenario 5
            temps_execution1_S5[n, c2, ii] = (end1 - start1) 
    for c3 in range(len(mise_en_correspondances3)):# for bf.L1, bf.L2 and bf.HAMMING matches
        mise_en_corresp3 = mise_en_correspondances3[c3]
        for i in range(len(Detecteurs)):    
            methode_keyPt = Detecteurs[i] # choose a detector from the "Detectors" list
            for j in range(len(Descripteurs)):
                methode_dscrpt = Descripteurs[j] # choose a descriptor from the "Descriptors" list 
                start2 = time.time() # start of the execution time for the Y method
                keypoints1   = methode_keyPt.detect(img1,None)
                keypoints2   = methode_keyPt.detect(img2,None)
                keypoints1   = methode_dscrpt.compute(img1, keypoints1)[0]# the keypoints of image 1 obtained by the Y method
                keypoints2   = methode_dscrpt.compute(img2, keypoints2)[0]# the keypoints of image 2 obtained by the Y method
                descriptors1 = methode_dscrpt.compute(img1, keypoints1)[1]# the descriptors of image 1 obtained by the Y method
                descriptors2 = methode_dscrpt.compute(img2, keypoints2)[1]# the descriptors of image 2 obtained by the Y method   
                # Calculation of the rate (%) of correctly matched homologous points and the number of homologous points by the Y method using the evaluation function of scenario 5
                Taux_stereo_rect2_S5[n, c3, i, j], Nbre_matches2_S5[n, c3, i, j] = evaluate_scenario_5_6_7(keypoints1, keypoints2, descriptors1, descriptors2, mise_en_corresp3)
                end2 = time.time() # end of the execution time for the Y method
                # Calculation of the execution time (in seconds) by the Y method using the evaluation function of scenario 5
                temps_execution2_S5[n, c3, i, j] = (end2 - start2) 
# ..........................................................................................................................
###############  Visualization of the results of scenario 5 by table (using the prettytable library)      
# ..........................................................................................................................

# The average of the results over all the images 
Nbre_matches1m_S5 = np.mean(Nbre_matches1_S5,0)
Nbre_matches2m_S5 = np.mean(Nbre_matches2_S5,0)
Taux_stereo_rect1m_S5 = np.mean(Taux_stereo_rect1_S5,0)
Taux_stereo_rect2m_S5 = np.mean(Taux_stereo_rect2_S5,0)
temps_execution1m_S5 = np.mean(temps_execution1_S5,0)
temps_execution2m_S5 = np.mean(temps_execution2_S5,0)

# concatination of the results in a vector for simplification of display in the table
Nbre_matches_Sén5 = np.concatenate((Nbre_matches1m_S5[0], Nbre_matches2m_S5[0][0], Nbre_matches2m_S5[0][1], Nbre_matches2m_S5[0][2], 
                                  Nbre_matches1m_S5[1], Nbre_matches2m_S5[1][0], Nbre_matches2m_S5[1][1], Nbre_matches2m_S5[1][2], 
                                  Nbre_matches2m_S5[2][0], Nbre_matches2m_S5[2][1], Nbre_matches2m_S5[2][2]), axis=0)
Taux_stereo_rect_Sén5 = np.concatenate((Taux_stereo_rect1m_S5[0], Taux_stereo_rect2m_S5[0][0], Taux_stereo_rect2m_S5[0][1], Taux_stereo_rect2m_S5[0][2], 
                                  Taux_stereo_rect1m_S5[1], Taux_stereo_rect2m_S5[1][0], Taux_stereo_rect2m_S5[1][1], Taux_stereo_rect2m_S5[1][2], 
                                  Taux_stereo_rect2m_S5[2][0], Taux_stereo_rect2m_S5[2][1], Taux_stereo_rect2m_S5[2][2]), axis=0)
temps_execution_Sén5 = np.concatenate((temps_execution1m_S5[0], temps_execution2m_S5[0][0], temps_execution2m_S5[0][1], temps_execution2m_S5[0][2], 
                                  temps_execution1m_S5[1], temps_execution2m_S5[1][0], temps_execution2m_S5[1][1], temps_execution2m_S5[1][2], 
                                  temps_execution2m_S5[2][0], temps_execution2m_S5[2][1], temps_execution2m_S5[2][2]), axis=0)

# The set of methods used to put them in caption
DetectDescript = list(['sift_bf.L1', 'akaze_bf.L1', 'orb_bf.L1', 'brisk_bf.L1', 'kaze_bf.L1', 'fast-freak_bf.L1', 'fast-brief_bf.L1', 
                  'fast-lucid_bf.L1', 'fast-latch_bf.L1', 'star-freak_bf.L1', 'star-brief_bf.L1', 'star-lucid_bf.L1', 'star-latch_bf.L1', 
                  'mser-freak_bf.L1', 'mser-brief_bf.L1', 'mser-lucid_bf.L1', 'mser-latch_bf.L1', 'sift_bf.L2', 'akaze_bf.L2', 'orb_bf.L2', 
                  'brisk_bf.L2', 'kaze_bf.L2', 'fast-freak_bf.L2', 'fast-brief_bf.L2', 'fast-lucid_bf.L2', 'fast-latch_bf.L2', 'star-freak_bf.L2', 
                  'star-brief_bf.L2', 'star-lucid_bf.L2', 'star-latch_bf.L2', 'mser-freak_bf.L2', 'mser-brief_bf.L2', 'mser-lucid_bf.L2', 
                  'mser-latch_bf.L2', 'fast-freak_bf.HAMG', 'fast-brief_bf.HAMG', 'fast-lucid_bf.HAMG', 'fast-latch_bf.HAMG', 'star-freak_bf.HAMG', 
                  'star-brief_bf.HAMG', 'star-lucid_bf.HAMG', 'star-latch_bf.HAMG', 'mser-freak_bf.HAMG', 'mser-brief_bf.HAMG', 'mser-lucid_bf.HAMG', 'mser-latch_bf.HAMG'])

# Initialization of the table, then give it a title, and add the column titles for our different results
x_Sén5 = PrettyTable()
x_Sén5.title = "Résultats du scénario 5"
x_Sén5.field_names = ['Méthode', "Temps d’exécution (s)", "Nombre de points/image", "Taux d’appariement correcte"]

# Organization of the rate of the correct matched points in decreasing order for the good visualization of the results   
Taux_sorted_Sén5 = sorted(Taux_stereo_rect_Sén5, reverse = True)

# Search for the coordinates of this organization to organize the results of the execution times and the number of homologous points detected in their vectors
coords_Sén5 = []
for i in range(len(Taux_sorted_Sén5)):
    coord = np.where(Taux_stereo_rect_Sén5 == Taux_sorted_Sén5[i])[0][0]
    coords_Sén5.append(coord)

# Displaying the results in the table in decreasing order of the rate of correct matched points
for i in coords_Sén5:
    x_Sén5.add_row([DetectDescript[i], round(temps_execution_Sén5[i], 3), round(Nbre_matches_Sén5[i], 3), round(Taux_stereo_rect_Sén5[i], 3)])
print(x_Sén5)
#.........................................................................................

################ Scenario 6 
N_S6 = 49 # Set of test images for scenario 6
# Initialization of the results matrices
Nbre_matches1_S6 = np.zeros((N_S6, len(mise_en_correspondances2), len(DetectDescript)))
Nbre_matches2_S6 = np.zeros((N_S6, len(mise_en_correspondances3), len(Detecteurs), len(Descripteurs)))
Taux_stereo_rect1_S6 = np.zeros((N_S6, len(mise_en_correspondances2), len(DetectDescript)))
Taux_stereo_rect2_S6 = np.zeros((N_S6, len(mise_en_correspondances3), len(Detecteurs), len(Descripteurs)))
temps_execution1_S6 = np.zeros((N_S6, len(mise_en_correspondances2), len(DetectDescript)))
temps_execution2_S6 = np.zeros((N_S6, len(mise_en_correspondances3), len(Detecteurs), len(Descripteurs)))

# for loop to compute rates (%) for scenario 6 images, matching, binary and non-binary methods
for n in range(N_S6):
    img1 = get_cam_Sén6(n)[0]# image 1 left of the stereo couple at t1
    img2 = get_cam_Sén6(n)[1]# image 2 right of the stereo pair at t2
    for c2 in range(len(mise_en_correspondances2)):# for bf.L1 and bf.L2 mappings (bf.HAMMING does not work for most non-binary methods)
        mise_en_corresp2 = mise_en_correspondances2[c2]
        for ii in range(len(DetectDescript)):
            methode = DetectDescript[ii]# choose a method from the "DetectDescript" list 
            start1 = time.time()  # start execution time for method X 
            keypoints11, descriptors11 = methode.detectAndCompute(img1, None)# the keypoints and descriptors of image 1 obtained by method X
            keypoints22, descriptors22 = methode.detectAndCompute(img2, None)# the keypoints and descriptors of image 2 obtained by method X
            # Calculation of the rate (%) of correctly matched homologous points and the number of homologous points by the X method using the evaluation function of scenario 6
            Taux_stereo_rect1_S6[n, c2, ii], Nbre_matches1_S6[n, c2, ii] = evaluate_scenario_5_6_7(keypoints11, keypoints22, descriptors11, descriptors22, mise_en_corresp2)  
            end1 = time.time() # end of execution time for the X method
            # Calculation of the execution time (in seconds) by the X method using the evaluation function of scenario 6
            temps_execution1_S6[n, c2, ii] = (end1 - start1) 
    for c3 in range(len(mise_en_correspondances3)):# for bf.L1, bf.L2 and bf.HAMMING matches
        mise_en_corresp3 = mise_en_correspondances3[c3]
        for i in range(len(Detecteurs)):    
            methode_keyPt = Detecteurs[i]# choose a detector from the "Detectors" list
            for j in range(len(Descripteurs)):
                methode_dscrpt = Descripteurs[j] # choose a descriptor from the "Descriptors" list 
                start2 = time.time() # start of the execution time for the method Y
                keypoints1   = methode_keyPt.detect(img1,None)
                keypoints2   = methode_keyPt.detect(img2,None)
                keypoints1   = methode_dscrpt.compute(img1, keypoints1)[0]# the keypoints of image 1 obtained by the Y method
                keypoints2   = methode_dscrpt.compute(img2, keypoints2)[0]# the keypoints of image 2 obtained by the Y method
                descriptors1 = methode_dscrpt.compute(img1, keypoints1)[1]# the descriptors of image 1 obtained by the Y method
                descriptors2 = methode_dscrpt.compute(img2, keypoints2)[1]# the descriptors of image 2 obtained by the Y method               
                # Calculation of the rate (%) of correctly matched homologous points and the number of homologous points by the Y method using the evaluation function of scenario 6
                Taux_stereo_rect2_S6[n, c3, i, j], Nbre_matches2_S6[n, c3, i, j] = evaluate_scenario_5_6_7(keypoints1, keypoints2, descriptors1, descriptors2, mise_en_corresp3)
                end2 = time.time() # end of execution time for the X method
                # Calculation of the execution time (in seconds) by the Y method using the evaluation function of scenario 6
                temps_execution2_S6[n, c3, i, j] = (end2 - start2) 
# ..........................................................................................................................
###############  Visualization of the results of scenario 6 by table (using the prettytable library)
# ..........................................................................................................................

# The average of the results over all the images 
Nbre_matches1m_S6 = np.mean(Nbre_matches1_S6,0)
Nbre_matches2m_S6 = np.mean(Nbre_matches2_S6,0)
Taux_stereo_rect1m_S6 = np.mean(Taux_stereo_rect1_S6,0)
Taux_stereo_rect2m_S6 = np.mean(Taux_stereo_rect2_S6,0)
temps_execution1m_S6 = np.mean(temps_execution1_S6,0)
temps_execution2m_S6 = np.mean(temps_execution2_S6,0)

# concatination of the results in a vector for simplification of display in the table
Nbre_matches_Sén6 = np.concatenate((Nbre_matches1m_S6[0], Nbre_matches2m_S6[0][0], Nbre_matches2m_S6[0][1], Nbre_matches2m_S6[0][2], 
                                  Nbre_matches1m_S6[1], Nbre_matches2m_S6[1][0], Nbre_matches2m_S6[1][1], Nbre_matches2m_S6[1][2], 
                                  Nbre_matches2m_S6[2][0], Nbre_matches2m_S6[2][1], Nbre_matches2m_S6[2][2]), axis=0)
Taux_stereo_rect_Sén6 = np.concatenate((Taux_stereo_rect1m_S6[0], Taux_stereo_rect2m_S6[0][0], Taux_stereo_rect2m_S6[0][1], Taux_stereo_rect2m_S6[0][2], 
                                  Taux_stereo_rect1m_S6[1], Taux_stereo_rect2m_S6[1][0], Taux_stereo_rect2m_S6[1][1], Taux_stereo_rect2m_S6[1][2], 
                                  Taux_stereo_rect2m_S6[2][0], Taux_stereo_rect2m_S6[2][1], Taux_stereo_rect2m_S6[2][2]), axis=0)
temps_execution_Sén6 = np.concatenate((temps_execution1m_S6[0], temps_execution2m_S6[0][0], temps_execution2m_S6[0][1], temps_execution2m_S6[0][2], 
                                  temps_execution1m_S6[1], temps_execution2m_S6[1][0], temps_execution2m_S6[1][1], temps_execution2m_S6[1][2], 
                                  temps_execution2m_S6[2][0], temps_execution2m_S6[2][1], temps_execution2m_S6[2][2]), axis=0)

# The set of methods used to put them in caption
DetectDescript = list(['sift_bf.L1', 'akaze_bf.L1', 'orb_bf.L1', 'brisk_bf.L1', 'kaze_bf.L1', 'fast-freak_bf.L1', 'fast-brief_bf.L1', 
                  'fast-lucid_bf.L1', 'fast-latch_bf.L1', 'star-freak_bf.L1', 'star-brief_bf.L1', 'star-lucid_bf.L1', 'star-latch_bf.L1', 
                  'mser-freak_bf.L1', 'mser-brief_bf.L1', 'mser-lucid_bf.L1', 'mser-latch_bf.L1', 'sift_bf.L2', 'akaze_bf.L2', 'orb_bf.L2', 
                  'brisk_bf.L2', 'kaze_bf.L2', 'fast-freak_bf.L2', 'fast-brief_bf.L2', 'fast-lucid_bf.L2', 'fast-latch_bf.L2', 'star-freak_bf.L2', 
                  'star-brief_bf.L2', 'star-lucid_bf.L2', 'star-latch_bf.L2', 'mser-freak_bf.L2', 'mser-brief_bf.L2', 'mser-lucid_bf.L2', 
                  'mser-latch_bf.L2', 'fast-freak_bf.HAMG', 'fast-brief_bf.HAMG', 'fast-lucid_bf.HAMG', 'fast-latch_bf.HAMG', 'star-freak_bf.HAMG', 
                  'star-brief_bf.HAMG', 'star-lucid_bf.HAMG', 'star-latch_bf.HAMG', 'mser-freak_bf.HAMG', 'mser-brief_bf.HAMG', 'mser-lucid_bf.HAMG', 'mser-latch_bf.HAMG'])

# Initialization of the table, then give it a title, and add the column titles for our different results
x_Sén6 = PrettyTable()
x_Sén6.title = "Résultats du scénario 6"
x_Sén6.field_names = ['Méthode', "Temps d’exécution (s)", "Nombre de points/image", "Taux d’appariement correcte"]

# Organization of the rate of the correct matched points in decreasing order for the good visualization of the results
Taux_sorted_Sén6 = sorted(Taux_stereo_rect_Sén6, reverse = True)

# Search for the coordinates of this organization to organize the results of the execution times and the number of homologous points detected in their vectors
coords_Sén6 = []
for i in range(len(Taux_sorted_Sén6)):
    coord = np.where(Taux_stereo_rect_Sén6 == Taux_sorted_Sén6[i])[0][0]
    coords_Sén6.append(coord)

# Display of the results in the table following the decreasing order of the rate of correct matched points 
for i in coords_Sén6:
    x_Sén6.add_row([DetectDescript[i], round(temps_execution_Sén6[i], 3), round(Nbre_matches_Sén6[i], 3), round(Taux_stereo_rect_Sén6[i], 3)])
print(x_Sén6)
#.........................................................................................

################ Scenario 7 
N_S7 = 48 # Set of test images for scenario 7
# Initialization of the result matrices
Nbre_matches1_S7 = np.zeros((N_S7, len(mise_en_correspondances2), len(DetectDescript)))
Nbre_matches2_S7 = np.zeros((N_S7, len(mise_en_correspondances3), len(Detecteurs), len(Descripteurs)))
Taux_stereo_rect1_S7 = np.zeros((N_S7, len(mise_en_correspondances2), len(DetectDescript)))
Taux_stereo_rect2_S7 = np.zeros((N_S7, len(mise_en_correspondances3), len(Detecteurs), len(Descripteurs)))
temps_execution1_S7 = np.zeros((N_S7, len(mise_en_correspondances2), len(DetectDescript)))
temps_execution2_S7 = np.zeros((N_S7, len(mise_en_correspondances3), len(Detecteurs), len(Descripteurs)))
# for loop to compute rates (%) for scenario 7 images, matches, binary and non-binary methods
for n in range(N_S7):
    img1 = get_cam_Sén7(n)[0]# image 1 left of the stereo couple at t1
    img2 = get_cam_Sén7(n)[1]# image 2 right of the stereo pair at t3
    for c2 in range(len(mise_en_correspondances2)):# for bf.L1 and bf.L2 mappings (bf.HAMMING does not work for most non-binary methods)
        mise_en_corresp2 = mise_en_correspondances2[c2]
        for ii in range(len(DetectDescript)):
            methode = DetectDescript[ii]# choose a method from the "DetectDescript" list 
            start1 = time.time() # start execution time for method X 
            keypoints11, descriptors11 = methode.detectAndCompute(img1, None)# the keypoints and descriptors of image 1 obtained by method X
            keypoints22, descriptors22 = methode.detectAndCompute(img2, None)# the keypoints and descriptors of image 2 obtained by method X
            # Computation of the rate (%) of correctly matched homologous points and the number of homologous points by the X method using the evaluation function of scenario 7
            Taux_stereo_rect1_S7[n, c2, ii], Nbre_matches1_S7[n, c2, ii] = evaluate_scenario_5_6_7(keypoints11, keypoints22, descriptors11, descriptors22, mise_en_corresp2)  
            end1 = time.time() # fin du temps d'execution pour la méthode X
            # Calculation of the execution time (in seconds) by the X method using the evaluation function of scenario 7
            temps_execution1_S7[n, c2, ii] = (end1 - start1) 
    for c3 in range(len(mise_en_correspondances3)):# for bf.L1, bf.L2 and bf.HAMMING matches
        mise_en_corresp3 = mise_en_correspondances3[c3]
        for i in range(len(Detecteurs)):    
            methode_keyPt = Detecteurs[i]# choose a detector from the "Detectors" list
            for j in range(len(Descripteurs)):
                methode_dscrpt = Descripteurs[j] # choose a descriptor from the "Descriptors" list 
                start2 = time.time() # start of the execution time for the method Y
                keypoints1   = methode_keyPt.detect(img1,None)
                keypoints2   = methode_keyPt.detect(img2,None)
                keypoints1   = methode_dscrpt.compute(img1, keypoints1)[0]# the keypoints of image 1 obtained by the Y method
                keypoints2   = methode_dscrpt.compute(img2, keypoints2)[0]# the keypoints of image 2 obtained by the Y method
                descriptors1 = methode_dscrpt.compute(img1, keypoints1)[1]# the descriptors of image 1 obtained by the Y method
                descriptors2 = methode_dscrpt.compute(img2, keypoints2)[1] # the descriptors of image 2 obtained by the Y method         
                # Computation of the rate (%) of correctly matched homologous points and the number of homologous points by the Y method using the evaluation function of scenario 7
                Taux_stereo_rect2_S7[n, c3, i, j], Nbre_matches2_S7[n, c3, i, j] = evaluate_scenario_5_6_7(keypoints1, keypoints2, descriptors1, descriptors2, mise_en_corresp3)
                end2 = time.time() 
                # Calculation of the execution time (in seconds) by the Y method using the evaluation function of scenario 7
                temps_execution2_S7[n, c3, i, j] = (end2 - start2) 
# ..........................................................................................................................
###############  Visualization of the results of scenario 7 by table (using the prettytable library)
# ..........................................................................................................................

# The average of the results over all the images 
Nbre_matches1m_S7 = np.mean(Nbre_matches1_S7,0)
Nbre_matches2m_S7 = np.mean(Nbre_matches2_S7,0)
Taux_stereo_rect1m_S7 = np.mean(Taux_stereo_rect1_S7,0)
Taux_stereo_rect2m_S7 = np.mean(Taux_stereo_rect2_S7,0)
temps_execution1m_S7 = np.mean(temps_execution1_S7,0)
temps_execution2m_S7 = np.mean(temps_execution2_S7,0)

# concatination of the results in a vector for simplification of display in the table
Nbre_matches_Sén7 = np.concatenate((Nbre_matches1m_S7[0], Nbre_matches2m_S7[0][0], Nbre_matches2m_S7[0][1], Nbre_matches2m_S7[0][2], 
                                  Nbre_matches1m_S7[1], Nbre_matches2m_S7[1][0], Nbre_matches2m_S7[1][1], Nbre_matches2m_S7[1][2], 
                                  Nbre_matches2m_S7[2][0], Nbre_matches2m_S7[2][1], Nbre_matches2m_S7[2][2]), axis=0)
Taux_stereo_rect_Sén7 = np.concatenate((Taux_stereo_rect1m_S7[0], Taux_stereo_rect2m_S7[0][0], Taux_stereo_rect2m_S7[0][1], Taux_stereo_rect2m_S7[0][2], 
                                  Taux_stereo_rect1m_S7[1], Taux_stereo_rect2m_S7[1][0], Taux_stereo_rect2m_S7[1][1], Taux_stereo_rect2m_S7[1][2], 
                                  Taux_stereo_rect2m_S7[2][0], Taux_stereo_rect2m_S7[2][1], Taux_stereo_rect2m_S7[2][2]), axis=0)
temps_execution_Sén7 = np.concatenate((temps_execution1m_S7[0], temps_execution2m_S7[0][0], temps_execution2m_S7[0][1], temps_execution2m_S7[0][2], 
                                  temps_execution1m_S7[1], temps_execution2m_S7[1][0], temps_execution2m_S7[1][1], temps_execution2m_S7[1][2], 
                                  temps_execution2m_S7[2][0], temps_execution2m_S7[2][1], temps_execution2m_S7[2][2]), axis=0)

# The set of methods used to put them in caption
DetectDescript = list(['sift_bf.L1', 'akaze_bf.L1', 'orb_bf.L1', 'brisk_bf.L1', 'kaze_bf.L1', 'fast-freak_bf.L1', 'fast-brief_bf.L1', 
                  'fast-lucid_bf.L1', 'fast-latch_bf.L1', 'star-freak_bf.L1', 'star-brief_bf.L1', 'star-lucid_bf.L1', 'star-latch_bf.L1', 
                  'mser-freak_bf.L1', 'mser-brief_bf.L1', 'mser-lucid_bf.L1', 'mser-latch_bf.L1', 'sift_bf.L2', 'akaze_bf.L2', 'orb_bf.L2', 
                  'brisk_bf.L2', 'kaze_bf.L2', 'fast-freak_bf.L2', 'fast-brief_bf.L2', 'fast-lucid_bf.L2', 'fast-latch_bf.L2', 'star-freak_bf.L2', 
                  'star-brief_bf.L2', 'star-lucid_bf.L2', 'star-latch_bf.L2', 'mser-freak_bf.L2', 'mser-brief_bf.L2', 'mser-lucid_bf.L2', 
                  'mser-latch_bf.L2', 'fast-freak_bf.HAMG', 'fast-brief_bf.HAMG', 'fast-lucid_bf.HAMG', 'fast-latch_bf.HAMG', 'star-freak_bf.HAMG', 
                  'star-brief_bf.HAMG', 'star-lucid_bf.HAMG', 'star-latch_bf.HAMG', 'mser-freak_bf.HAMG', 'mser-brief_bf.HAMG', 'mser-lucid_bf.HAMG', 'mser-latch_bf.HAMG'])

# Initialization of the table, then give it a title, and add the column titles for our different results
x_Sén7 = PrettyTable()
x_Sén7.title = "Résultats du scénario 7"
x_Sén7.field_names = ['Méthode', "Temps d’exécution (s)", "Nombre de points/image", "Taux d’appariement correcte"]

# Organization of the rate of the correct matched points in decreasing order for the good visualization of the results
Taux_sorted_Sén7 = sorted(Taux_stereo_rect_Sén7, reverse = True)

# Search for the coordinates of this organization to organize the results of the execution times and the number of homologous points detected in their vectors
coords_Sén7 = []
for i in range(len(Taux_sorted_Sén7)):
    coord = np.where(Taux_stereo_rect_Sén7 == Taux_sorted_Sén7[i])[0][0]
    coords_Sén7.append(coord)

# Display of the results in the table following the decreasing order of the rate of correct matched points  
for i in coords_Sén7:
    x_Sén7.add_row([DetectDescript[i], round(temps_execution_Sén7[i], 3), round(Nbre_matches_Sén7[i], 3), round(Taux_stereo_rect_Sén7[i], 3)])
print(x_Sén7)

###################################################         END      ###################################################################################