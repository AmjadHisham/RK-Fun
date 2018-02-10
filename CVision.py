import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import sys
import sklearn.cluster
import DSP


def cv2_color2gray(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def is_grayscale(img):
    if len(img.shape[:])==2:
        return True
    else:
        return False

def cv2_to_plt(img):
    
    if is_grayscale(img):
        plt.imshow(img,cmap='gray')
        plt.show()
        plt.close()
    else:
        b= img[:,:,0]
        g= img[:,:,1]
        r= img[:,:,2]
        img = cv2.merge([r,g,b])
        plt.imshow(img)
        plt.show()
        plt.close()
    
def cv2_resize(img,shape):
    rows,cols = shape
    return cv2.resize(img,(cols,rows))


def kmeans_color_clustering(image,n):
    sqrt_n = int(np.sqrt(n))
    assert sqrt_n == np.sqrt(n)
    three_plane_color = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = sklearn.cluster.KMeans(n_clusters = n)
    clt.fit(three_plane_color)
    cluster =  clt.cluster_centers_.reshape(n,3)
    
    rows,cols = 600,600
    img = np.zeros((rows,cols,3), np.uint8)

    topleft = np.array([0,0])
    bottomright = np.array([rows//sqrt_n,cols//sqrt_n])
    k=0
    for i in range(0,sqrt_n):
        for j in range(0,sqrt_n):
            img = cv2.rectangle(img,tuple(reversed(topleft)),tuple(reversed(bottomright)),(cluster[k][0],cluster[k][1],cluster[k][2]),-1)
            k+=1
            #print(topleft,bottomright)
            topleft+=np.array([0,cols//sqrt_n])
            bottomright+=np.array([0,cols//sqrt_n])
        topleft+=np.array([rows//sqrt_n,0])
        topleft[1]=0
        bottomright+=np.array([rows//sqrt_n,0])
        bottomright[1]=rows//sqrt_n

    output = cv2.resize(image,(rows,cols))
    output = np.hstack((output,img))
    plt.figure(figsize=(7,5))
    cv2_to_plt(output)
    return cluster


def color_segmentation(image,centroid,threshold):

    B = image[:,:,0].astype(np.int16) 
    G = image[:,:,1].astype(np.int16) 
    R = image[:,:,2].astype(np.int16) 

    B = np.subtract(B,centroid[0]) 
    B = np.multiply(B,B) 

    R = np.subtract(R,centroid[2]) 
    R = np.multiply(R,R) 

    G = np.subtract(G,centroid[1]) 
    G = np.multiply(G,G) 

    C = np.add(B,G) 
    C = np.add(C,R) 
    C = np.sqrt(C) 

    C_ = C[np.where(C<threshold)] 
    Cs = np.in1d(C.ravel(), C_).reshape(C.shape)
    out = np.zeros_like(image) 
    out[Cs] = image[Cs]
    plt.figure(figsize=(7,5))
    cv2_to_plt(out)
    

def get_object(image,canny_param_1=35,canny_param_2=135,min_area=500,min_perimeter=60):
    
    def subtract_background(image,y,x,w,h,approx):
        mask = np.zeros((image.shape[0], image.shape[1]))
        cv2.fillConvexPoly(mask, approx, 1)
        mask = mask.astype(np.bool)
        out = np.zeros_like(image)
        out[mask] = image[mask]
        box = cv2.rectangle(image.copy(),(y,x),(y+w,x+h),color=(20,100,20),thickness=3)
        plt.figure(figsize=(14,7))
        cv2_to_plt(np.hstack((image,box,out)))

    gray=image.copy()
    if len(image.shape[:])==3:
        print("converting image to grayscale")
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        
    edged = cv2.Canny(gray, canny_param_1, canny_param_2)

    _, cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
    
    package = {}
    keys = ["top_left_pt","width_height","contour_area","perimeter","convex","rect_area"]
    for key in keys:
        package[key] = []
    
    for c in cnts:
        marker = cv2.minAreaRect(c)
        y,x,w,h = cv2.boundingRect(c)
        M = cv2.moments(c)
        area = cv2.contourArea(c)
        object_area = marker[1][0]*marker[1][1]
        perimeter = cv2.arcLength(c,True)
        convex = cv2.isContourConvex(c)
        epsilon = 0.01*cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,epsilon,True)
        image_area = image.shape[1]*image.shape[0]

        print("top left point: ",(x,y),"width and height: ",(w,h) ,"contour area: ", area," perimeter: ",
              perimeter," is convex:",convex," square area: ",object_area)
        if area>=min_area and perimeter>=min_perimeter:
            subtract_background(image,y,x,w,h,approx)
            package["top_left_pt"].append((x,y))
            package["width_height"].append((w,h))
            package["contour_area"].append(area)
            package["perimeter"].append(perimeter)
            package["convex"].append(convex)
            package["rect_area"].append(object_area)
            
            
    return package


    
def custom_2d_kernel_conv(image, kernel, test=False, bordered=True):
    
    
    if len(image.shape[:])==3:
        print("converting image to grayscale")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    
    

    assert len(image.shape[:])==len(kernel.shape[:])
    
    
    def sliding_2d_conv(img, kernel):
    
        m,n=img.shape[:2]
        j,k = kernel.shape[:2]
        pad = (k-1)//2
        out = np.zeros((m,n),np.float64)

        window_size = (k-1)//2

        for j in range(m):
            xi = j-window_size
            xf = j+window_size+1

            if j < window_size:
                continue
            elif (m-j)< window_size+1:
                continue
            for i in range(n):
                yi = i-window_size
                yf = i+window_size+1
                if i < window_size:
                    continue
                elif (n-i)< window_size+1:
                    continue

                out[j:j+1,i:i+1] = np.sum(img[xi:xf,yi:yf].copy()*kernel)
                
        return  out

    def img_border(img,kernel):
        m,n = img.shape[:]
        j,k = kernel.shape[:]
        pad = (k-1)//2
        out = np.zeros((m+2*pad,n+2*pad),np.float64)
        out[pad:m+pad,pad:n+pad] = img
        
        #plt.imshow(out,cmap = 'gray')
        #plt.show()
        return out
    
    def img_deborder(img):
        m,n = img.shape[:]
        j,k = kernel.shape[:]
        pad = (k-1)//2
        return img[pad:m-pad,pad:n-pad]

    
    if bordered:
        bordered = img_border(image,kernel)
        output = sliding_2d_conv(bordered, kernel)
        output = img_deborder(output)
    else:
        output = sliding_2d_conv(image, kernel)
    
    if test:
        print("kernel")
        print(kernel)
        print("bordered image: top left")
        print(bordered[:3,:3])
        print("bordered image: bottom right")
        print(bordered[-3:,-3:])
        print("output image: top left")
        print(output[:3,:3])
        print("output image: bottom right")
        print(output[-3:,-3:])
    
    return output

def sliding_phase_correlation(img, kernel,min_corr=0.9,plot=True):
    
    def img_cross_correlate(img1,img2):
    
        unfiltered1 = DSP.get_frequency_domain(img1.flatten())

        unfiltered2 = DSP.get_frequency_domain(img2.flatten())

        return unfiltered1[:,1]*np.conjugate(unfiltered2[:,1])
    
    if len(img.shape[:])==3:
        print("converting image to grayscale")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    if len(kernel.shape[:])==3:
        print("converting image to grayscale")
        kernel = cv2.cvtColor(kernel, cv2.COLOR_BGR2GRAY)
        
    
    m,n=img.shape[:]
    row_size,col_size = kernel.shape[:]
    
    c = []
    pts1 = []
    pts2 = []
    j=0
    while True:
        xi =j
        xf = xi+row_size
       
        i=0
        while True:
            yi = i 
            yf = yi + col_size

            cross = img_cross_correlate(img[xi:xf,yi:yf].copy(),kernel)

            norm_cross = np.fft.ifft(cross/np.absolute(cross))
            
            corr = np.absolute(np.max(norm_cross))
            
            if corr>min_corr:
                pt1 = (yi,xi)
                pt2 = (yf,xf)
                pts1.append(pt1)
                pts2.append(pt2)
                c.append(corr)
                
            if corr>min_corr and plot:
                print("correlation: ",corr)
                print("row range", xi,":",xf)
                print("col range", yi,":",yf)
                cv2_to_plt(img[xi:xf,yi:yf])
                

            i+=1
            if yf>=n:
                break
        j+=1
        if xf>=m:
            break

        time.sleep(0.01)
    return c,pts1,pts2


def four_quadrant_indentifier(frame,plot=False):
    
    gray = frame
    if len(frame.shape[:])==3:
        print("converting image to grayscale")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    im_arr=gray
    im_height=im_arr.shape[0]
    im_width=im_arr.shape[1]
    intensity_overall=np.mean(im_arr)
    y=im_arr.shape[0]//2
    x=im_arr.shape[1]//2
    
        
    sub_im1 =  im_arr[0:y,0:x]
    upperleft = np.mean(sub_im1 )

    sub_im2 =  im_arr[0:y,x:x*2]
    upperright = np.mean(sub_im2 )

    sub_im3 =  im_arr[y:y*2,0:x]
    lowerleft = np.mean(sub_im3 )

    sub_im4 =  im_arr[y:y*2,x:x*2]
    lowerright = np.mean(sub_im4 )
    
    if plot:
        c = (20, 20, 190)
        cv2_to_plt(cv2.line(cv2.line(frame,(x,0),(x,y*2),color=c),(0,y),(x*2,y),color=c,thickness=2))
    
    return upperright,upperleft,lowerleft,lowerright


def im_stitcher(imp1, imp2, withTransparency=True, plot=False, warp_threshold=10000):
    
    #Read image1
    image1 = imp1

    #Read image2
    image2 = imp2

    
    im2=cv2.resize(image2,(1024,1024),interpolation=cv2.INTER_AREA)
    im1=cv2.resize(image1,(1024,1024),interpolation=cv2.INTER_AREA)

    img2Gray = im2
    img1Gray = im1
    
    if len(im2.shape[:])==3:
        #print("converting image to grayscale")
        img2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        
    if len(im1.shape[:])==3:
        #print("converting image to grayscale")
        img1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

    #use BRISK to create keypoints in each image
    brisk = cv2.BRISK_create()

    kp1, des1 = brisk.detectAndCompute(img1Gray,None)
    kp2, des2 = brisk.detectAndCompute(img2Gray,None)
    
    # use BruteForce algorithm to detect matches among image keypoints 
    dm = cv2.DescriptorMatcher_create("BruteForce-Hamming")
    
    matches = dm.knnMatch(des1,des2, 2)
    matches_ = []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * 0.7:
            matches_.append((m[0].trainIdx, m[0].queryIdx))
    
    kp1_ = np.float32([kp1[m[1]].pt for m in matches_]).reshape(-1,1,2)
    kp2_ = np.float32([kp2[m[0]].pt for m in matches_]).reshape(-1,1,2)

    try: 
        H, mask = cv2.findHomography(kp2_,kp1_, cv2.RANSAC, 1)
        h1,w1 = im1.shape[:2]
        h2,w2 = im2.shape[:2]

        pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
        pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)

        pts2_ = cv2.perspectiveTransform(pts2, H)
        pts = np.concatenate((pts1, pts2_), axis=0)
        #print("Pts: ",pts)
        [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)

        t = [-xmin,-ymin]

        #print(xmax-xmin, ymax-ymin)
        #print((pts1-pts2_))
        
        print("Warp Threshold",np.sum(abs((pts1-pts2_))))
        if(np.sum(abs((pts1-pts2_)))>warp_threshold):
            print(np.sum(abs((pts1-pts2_))))
            raise RuntimeError('The dimensions of projected image exceed predefined dimensions')
        Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]])

        #warp the colour version of image2
        im = cv2.warpPerspective(img2Gray, Ht.dot(H), (xmax-xmin, ymax-ymin))
        im_warp=im
        #overlay colur version of image1 to warped image2
        if withTransparency == True:
            h3,w3 = im.shape[:2]
            bim = np.zeros((h3,w3), np.uint8)
            bim[t[1]:h1+t[1],t[0]:w1+t[0]] = img1Gray


            imGray = im
            imColor = cv2.applyColorMap(imGray, cv2.COLORMAP_JET)
            bimGray = bim
            bimColor = cv2.applyColorMap(bimGray, cv2.COLORMAP_JET)

            im =np.subtract(bimGray,cv2.multiply(imGray,0.7))
            #im = cv2.addWeighted(im,0.5,bim,0.5,0)
        else:
            im[t[1]:h1+t[1],t[0]:w1+t[0]] = im1

        im3 = cv2.resize(im,(1024,1024),interpolation=cv2.INTER_AREA)
        im3=im
        
        if plot:
            print("Original")
            plt.imshow(img1Gray)
            plt.show()  
            print("Comparision")
            plt.imshow(img2Gray)
            plt.show()  
            print("Warped")
            plt.imshow(im_warp)
            plt.show()
            print("Subtracted Image")
            plt.imshow(im3[t[1]:h1+t[1],t[0]:w1+t[0]])
            plt.show()
        
        if(np.mean(im3[t[1]:h1+t[1],t[0]:w1+t[0]])<0.8*np.mean(im1)):
            return img1Gray,im_warp[t[1]:h1+t[1],t[0]:w1+t[0]],True
        else:
            return img1Gray,im_warp[t[1]:h1+t[1],t[0]:w1+t[0]],False


    except Exception as e:
        print(e)
        pass

    return np.zeros((1024,1024)),np.zeros((1024,1024)),False



def remove_n_vertical_seams(img,n,plot=False):
    
    
    def compute_energy_matrix(img):
        if ~is_grayscale(img):
            img = cv2_color2gray(img)

        sobel_x  =cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
        sobel_y  =cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)

        abs_sobel_x = cv2.convertScaleAbs(sobel_x)
        abs_sobel_y = cv2.convertScaleAbs(sobel_y)
    
        return cv2.addWeighted(abs_sobel_x,0.5,abs_sobel_y,0.5,0)

    rows,cols = img.shape[:2]
    img_seams = img.copy()
    seams = np.zeros((img.shape[0],n))
    
    while n>0:
        n=n-1
        rows,cols = img.shape[:2]
        energy = compute_energy_matrix(img)
        
        dist_to = np.zeros((rows,cols))+sys.maxsize

        dist_to[0,:] = np.zeros(cols)
        edge_to = np.zeros((rows,cols))


        for row in range(rows-1):
            for col in range(cols):

                x = row+1
                start = col-1
                end = col+1
                
                #make sure that the start column is 0 when the column is 0
                #make sure that the end columns is col when the column is cols-1
                if col==0:
                    start = col
                elif col==cols-1:
                    end = col
                    
                for y in range(start,end+1):
                    if (dist_to[x,y]>dist_to[row,col]
                        +energy[x,y]): 
                        dist_to[x,y] = dist_to[row,col]+energy[x,y]
                        edge_to[x,y] = col - y
    
        #backtrack edge matrix and calculate seam path
        seams[rows-1] = np.argmin(dist_to[rows-1,:])
        for i in (x for x in reversed(range(rows)) if x>0):
            seams[i-1,n] = seams[i,n] + edge_to[i, int(seams[i,n])]

        #remove seams
        for row in range(rows):
            for col in range(int(seams[row,n]),cols-1):
                img[row,col] = img[row,col+1]
        img=img[:,0:cols-1]
        
    if plot:
        
        img_seams = np.column_stack((img_seams,np.zeros((rows,n+1,3),np.uint8)))
        for seam in seams.T[::-1]:
            for i,j in enumerate(seam):
                img_seams[i,int(j)+1] = img_seams[i,int(j)]
                img_seams[i,int(j)] = (0,255,0)
        plt.figure(figsize=(7,5))
        cv2_to_plt(img_seams)
    
        
    return img,seams

def im_flip(img1,flip=0):
    h1,w1 = img1.shape[:2]
    src_pts = np.float32([[0,0],[w1-1,0],[0,h1-1]])
    dst_pts = np.float32([[w1-1,0],[0,0],[w1-1,h1-1]])
    affine_matrix = cv2.getAffineTransform(src_pts,dst_pts)
    img2 = cv2.warpAffine(img1,affine_matrix,(w1,h1))
    if flip:
        return img2
    else:
        return img1

def im_warp_left(img1,PUR,PLL):
    h1,w1 = img1.shape[:2]
    src_pts = np.float32([[0,0],[w1-1,0],[0,h1-1]])
    dst_pts = np.float32([[0,0],[int(PUR*(w1-1)),0],[int(PLL*(w1-1)),h1-1]])
    affine_matrix = cv2.getAffineTransform(src_pts,dst_pts)
    img2= cv2.warpAffine(img1,affine_matrix,(w1,h1))
    return img2

def im_warp_right(img1,PUL,PLL):

    h1,w1 = img1.shape[:2]
    src_pts = np.float32([[0,0],[w1-1,0],[0,h1-1]])
    dst_pts = np.float32([[int(PUL*(w1-1)),0],[(w1-1),0],[int(PLL*(w1-1)),h1-1]])
    affine_matrix = cv2.getAffineTransform(src_pts,dst_pts)
    img2 = cv2.warpAffine(img1,affine_matrix,(w1,h1))
    return img2

