
# coding: utf-8

# # Detecting, extracting and matching features

# In[4]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import random


# In[59]:


name = input("Enter image name's first number(1,2,3):")

ref_img = cv2.imread(name+"_0.JPG")
ref_depth = cv2.imread(name+"_0d.JPG",0)

source_img = cv2.imread(name+"_1.JPG")


m,n,_ = ref_img.shape


# In[40]:


# Keypoint Detection using SIFT
def keypoints(IM):  
    sift = cv2.xfeatures2d.SIFT_create()

    KP,DES = sift.detectAndCompute(IM,None)

    return KP, DES


# In[41]:


# Matching Keypoints using Lowe's Ratio Test
def matching(DES_1,DES_2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(DES_1,DES_2, k=2)
    match = []
    for M,N in matches:
        if M.distance < 0.75*N.distance:
            match.append([M])
    return match


# # Quantize depth level

# In[42]:


# Finding Matching Keypoints between the two images
ref_kp,ref_des = keypoints(ref_img)
source_kp,source_des = keypoints(source_img)
good = matching(ref_des,source_des)

# Depth is divided into 5 levels
matches = [[],[],[],[],[]]

for i in range(len(good)):
    ref_coord = list(ref_kp[good[i][0].queryIdx].pt)
    x = ref_coord[0]
    y = ref_coord[1]
    d = ref_depth[int(y),int(x)]
    
    # Store the matched points using depth information
    matches[int(np.floor(d/52))].append(good[i])


# In[43]:


# The following lines check whether atleast 20 matched keypoints are present at each depth level
# If not they are replaced by the keypoints of nearest level which have more than 20 matches

for i in range(5):
    if(len(matches[i]) < 20):
        flag = 0
        for j in range(i+1,5):
            if(len(matches[j])>20):
                matches[i] = matches[j]
                
                flag = 1
                break
        if(flag != 1):
            for j in range(i-1,-1,-1):
                if(len(matches[j])>20):
                    matches[i] = matches[j]
                    break


# # RANSAC Algorithm

# In[44]:


# Nullspace of a matrix
def nullspace(A, atol=1e-13, rtol=0):
    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns


# In[45]:


# Homography matrix found using DLT
def Homography(coord_1,coord_2):
    H = []
    A = np.zeros((2*len(coord_1),9))
    count = 0
    
    # Array to normalise the coordinates
    T = np.array([[2/n, 0, -1],[0, 2/m, -1],[0,0,1]])
    
    for i in range(len(coord_1)):
        coord_1_norm = T@coord_1[i]
        coord_2_norm = T@coord_2[i]
        x_1 = coord_1_norm[0]
        y_1 = coord_1_norm[1]
        x_2 = coord_2_norm[0]
        y_2 = coord_2_norm[1]
        A[count,:] = [x_1, y_1, 1, 0, 0, 0, -x_1*x_2,-y_1*x_2,-x_2]
        A[count+1,:] = [0, 0, 0, x_1, y_1, 1, -x_1*y_2,-y_1*y_2,-y_2] 
        count += 2
        
    if(len(coord_1)==4):
        h = nullspace(A)
    else:
        U,S,Vt = np.linalg.svd(A)
        
        # Taking the last row of Vt(V transpose)
        h = Vt[-1,:].reshape(9,1)
    
    # It may happen that 3 out of 4 random points from ransac are collinear and A matrix would not be full rank
    # So this condition is put
    if(h.shape[1] == 1):
        H = h.reshape((3,3))
        
        # Returns unnormalised Homography Matrix
        H = np.linalg.inv(T)@H@T
        return H/H[2,2]
    else:
        return H
        
        


# In[46]:


# RANSAC algorithm
def ransac(KP_1,KP_2,match,threshold):
    
    # Max number of inliers
    maxx = np.NINF

    for i in range(1000):
        # Randomly Select 4 points
        rand_idx = random.sample(range(1,len(match)),4)
        img_1_coord = []
        img_2_coord = []
        for j in rand_idx:
            img_1_coord_ = list(KP_1[match[j][0].queryIdx].pt)
            img_2_coord_ = list(KP_2[match[j][0].trainIdx].pt)
            img_1_coord_.append(1)
            img_2_coord_.append(1)
            img_1_coord_ = np.array(img_1_coord_)
            img_2_coord_ = np.array(img_2_coord_)
            img_1_coord.append(img_1_coord_)
            img_2_coord.append(img_2_coord_)
        
        # Find Homography of 4 points
        H = Homography(img_1_coord,img_2_coord)
        
        if(len(H) != 0):
            inliers = 0
            img1_inlier_coord = []
            img2_inlier_coord = []
            for j in range(len(match)):
                if(j not in rand_idx):
                    img_1_coord = list(KP_1[match[j][0].queryIdx].pt)
                    img_2_coord = list(KP_2[match[j][0].trainIdx].pt)
                    img_1_coord.append(1)
                    img_2_coord.append(1)
                    img_1_coord = np.array(img_1_coord)
                    img_2_coord = np.array(img_2_coord)
                    
                    # Finding transformed coordinates using H
                    transform_coord = H @ img_1_coord
                    transform_coord /= transform_coord[2]
                    
                    # If the distance of two points is less than threshold, store them as inliers 
                    if(np.linalg.norm((transform_coord)-img_2_coord) < threshold):
                        img1_inlier_coord.append(img_1_coord)
                        img2_inlier_coord.append(img_2_coord)
                        inliers += 1 
            # If inliers found in this iteration are more than previous all then update final_coord
            if(inliers>maxx):
                final_coord_1 = img1_inlier_coord
                final_coord_2 = img2_inlier_coord
                maxx = inliers
    # Return Homography found using inliers
    H = Homography(final_coord_1,final_coord_2)
    return H
    
                


# In[47]:


# H_matrix[i] is mapping of the ith depth level
# Threshold is set to 10
H_matrix = []
for i in range(5):
    H_matrix.append(ransac(ref_kp,source_kp,matches[i],10))
    print("Homography for level "+ str(i+1))
    print(H_matrix[i])


# # Final Panorama

# In[48]:


# Returns Gaussian Weights for blending
def weights(shape_):
    rows = shape_[0]
    cols = shape_[1]
    
    W = np.zeros((rows,cols))
    
    sigma = np.ceil(min(rows,cols)/6)
    temp_r = int(np.floor(rows/2))
    temp_c = int(np.floor(cols/2))
    temp_r_neg = -int(np.floor(rows/2))
    temp_c_neg = -int(np.floor(cols/2))
    if(rows%2 == 0):
        temp_r_neg += 1
    if(cols%2 == 0):
        temp_c_neg += 1
    for i in range(temp_r_neg,temp_r+1):
        for j in range(temp_c_neg,temp_c+1):
            W[i-temp_r_neg,j-temp_c_neg] = np.exp(-(i**2+j**2)/(2*(sigma**2)));
    return W


# In[49]:


# Boundary of the final panorama
def boundary(shape_,homographies):
    minn_x = np.Inf
    minn_y = np.Inf
    maxx_x = np.NINF
    maxx_y = np.NINF
    
    extremas = [[0,0,1],[n,0,1],[0,m,1],[n,m,1]]
    for i in range(len(homographies)):
        for j in range(len(extremas)):
            point = homographies[i]@extremas[j]
            point /= point[2]

            x = point[0]
            y = point[1]

            if(x > maxx_x):
                maxx_x = x
            if(x < minn_x):
                minn_x = x
            if(y > maxx_y):
                maxx_y = y
            if(y < minn_y):
                minn_y = y
    if(n > maxx_x):
        maxx_x = n
    if(0 < minn_x):
        minn_x = 0
    if(m > maxx_y):
        maxx_y = m
    if(0 < minn_y):
        minn_y = 0

    return int(minn_x),int(minn_y),int(maxx_x),int(maxx_y)
    


# In[50]:


min_x, min_y, max_x, max_y = boundary((m,n),H_matrix)

out = np.zeros((-min_y+max_y+10,-min_x+max_x+10,3))

out[-min_y+5:-min_y+m+5,-min_x+5:-min_x+n+5,:] = source_img

w = weights((m,n))

# Warping Reference image to source image

for y in range(m):
    for x in range(n):
        
        # Calculate which Homography matrix to use
        d = ref_depth[int(y),int(x)]
        H = H_matrix[int(np.floor(d/52))]
        
        new_point = H@[x,y,1]
        new_point /= new_point[2]
        
        #Transformed points
        x_ = int(new_point[0])
        y_ = int(new_point[1])

        # Finding common region for blending
        # Same pixel value is copied in a neighborhood of the transformed coordinates
        if(0<x_<n and  0<y_<m):
            out[y_-min_y+5-1:y_-min_y+5+2,x_-min_x+5-1:x_-min_x+5+2,:] = (w[y,x]*ref_img[y,x,:]+w[y_,x_]*source_img[y_,x_])/(w[y,x]+w[y_,x_])
        else:
            out[y_-min_y+5-1:y_-min_y+5+2,x_-min_x+5-1:x_-min_x+5+2,:] = ref_img[y,x,:]

plt.imshow(np.uint8(out)[:,:,::-1])
plt.title("Panorama using depth")
plt.show()
# Uncomment below to save the image
# cv2.imwrite("Panoramawithdepth_"+name+".jpg",np.uint8(out))


# # Panorama using single Homography

# In[52]:


H = ransac(ref_kp,source_kp,good,10)
print("Single Homography Matrix")
print(H)

# Same piece of code as above
min_x, min_y, max_x, max_y = boundary((m,n),[H])

out = np.zeros((-min_y+max_y+10,-min_x+max_x+10,3))

out[-min_y+5:-min_y+m+5,-min_x+5:-min_x+n+5,:] = source_img

w = weights((m,n))

for y in range(m):
    for x in range(n):

        new_point = H@[x,y,1]
        new_point /= new_point[2]
        x_ = int(new_point[0])
        y_ = int(new_point[1])

        
        if(0<x_<n and  0<y_<m):
            out[y_-min_y+5-1:y_-min_y+5+2,x_-min_x+5-1:x_-min_x+5+2,:] = (w[y,x]*ref_img[y,x,:]+w[y_,x_]*source_img[y_,x_])/(w[y,x]+w[y_,x_])
        else:
            out[y_-min_y+5-1:y_-min_y+5+2,x_-min_x+5-1:x_-min_x+5+2,:] = ref_img[y,x,:]

plt.imshow(np.uint8(out)[:,:,::-1])
plt.title("Panorama without depth")
plt.show()

# Uncomment below to save the image
# cv2.imwrite("Panoramawithoutdepth_"+name+".jpg",out)

