class CameraImages():
    
    def __init__(self, file_dict,process_layer_fcn1,process_layer_fcn2=None,process_layer_fcn3=None):
        self.dict_ = file_dict
        self.ret_ = [] 
        self.cat_ = []
        self.indx_ = []
        self.avg_phasecorr = 0.0
        self.img_indx = 0
        self.process_layer_fcn1 = process_layer_fcn1
        self.process_layer_fcn2 = process_layer_fcn2
        self.process_layer_fcn3 = process_layer_fcn3
    

    def get_layers(self,sub):
        layer1 = self.process_layer_fcn1(sub)
        layer2 = self.process_layer_fcn2(sub)
        layer3 = self.process_layer_fcn3(sub)
        return np.dstack((layer1,layer2,layer3))

    def subsections(self,img):
        
        rows,cols = img.shape[:]
#         print(rows,cols)
#         assert rows/100 == rows//100

        count = 0
        row = 0
        while row+100<rows:
            col = 0
            while col+100<cols:

                variance_check = np.var(img[row:row+100,col:col+100])>5
                if (np.random.randint(0,10) == 0) & variance_check:
                    sub = img[row:row+100,col:col+100]
                    
#                     if self.process_layer_fcn2==None | self.process_layer_fcn3==None:
#                         layer_stack = self.process_layer_fcn1(sub)
#                     else:
                    layer_stack = self.get_layers(sub)
                    if ~np.any(np.sum(layer_stack,axis=(0,1))==0):
                        self.ret_.append(layer_stack)
                        self.indx_.append(self.img_indx)
                        count+=1
                    else:
                        pass
#                         print(np.sum(layer_stack,axis=(0,1)))

                if count>=3:
                    break

                col = col+100
                
            row = row+100
            

            

            


    def make_tiles(self):
        
        values = self.dict_
        for x in values:
            img = cv2.imread(x)
            img = CVision.cv2_color2gray(img)
            self.subsections(img)
            self.img_indx+=1
            
#         return self.ret_

    def simulate(self):
        img = np.mean(np.array(self.ret_),axis=0)
        plt.figure(figsize=(10,10))
        plt.imshow(img)
        plt.show()
    
    
    def img_cross_correlate(self,img1,img2):
    
        unfiltered1 = DSP.get_frequency_domain(img1.flatten())

        unfiltered2 = DSP.get_frequency_domain(img2.flatten())

        return unfiltered1[:,1]*np.conjugate(unfiltered2[:,1])
    
    def phase_correlate(self):
        
        c = 0
        for i in range(len(self.ret_)):
            for j in range(len(self.ret_)):
                if i!=j:
                    img1 = self.ret_[i]
                    img2 = self.ret_[j]
                    img1 = img1.astype(np.float)
                    img2 = img2.astype(np.float)

                    cross = self.img_cross_correlate(img1,img2)

                    norm_cross = np.fft.ifft(cross/np.absolute(cross))

                    corr = np.absolute(np.max(norm_cross))
                    
                    if ~(corr>=0) | np.isnan(corr):
                        corr = 0

                    if c==0:
                        self.avg_phasecorr = corr
                    elif c>0:
                        self.avg_phasecorr = (self.avg_phasecorr*c/(c+1) + corr/(c+1))
                        
                    c+=1
    
                    break
            break


                
            
    def category(self,val):
        self.cat_ = np.repeat(val,len(self.ret_))
        
        
        
        
        
class ProcessingFunctions():

    def __init__(self, k=3,peak_num=1):
        self.k = k
        self.peak_num = peak_num
        
        
    def get_1std(self,mat):
        return np.mean(mat)+1*np.std(mat)
        
        
    def histogram_mask_mean_counts(self,img):
        img = cv2.equalizeHist(img)

        hist=np.histogram(img.ravel(),256,[0,256])
        hist = [hist[0], 0.5*(hist[1][1:]+hist[1][:-1]) - 0.5]

        mask = hist[1][(hist[0]-np.median(hist[0]))<=30]

#         mask = hist[1][hist[1]==0]
        table = []
        for i in np.arange(0,256).astype(np.uint8):
            if i in mask:
                table.append(i)
#                 table.append(((i / 255.0) ** (1/0.5)) * 255)
            else:
                table.append(0)
        table = np.array(table)
        # table = np.array([((i / 255.0) ** invGamma) * 255
        #         for i in np.arange(0, 256)]).astype("uint8")

        img =  cv2.LUT(img,table)
#         if np.sum(img) == 0:
#             print('error')
#         plt.figure(figsize=(10,10))
#         plt.imshow(img)
#         plt.show()

        return img

    def histogram_mask_dark_pixels(self,img):
        img = cv2.equalizeHist(img)

        hist=np.histogram(img.ravel(),256,[0,256])
        hist = [hist[0], 0.5*(hist[1][1:]+hist[1][:-1]) - 0.5]

        mask = hist[1][hist[1]<10]
#         mask = hist[1][hist[1]==0]
        table = []
        for i in np.arange(0,256).astype(np.uint8):
            if i in mask:
#                 table.append(i)
                table.append(((i / 255.0) ** (1/0.5)) * 255)
            else:
                table.append(0)
        table = np.array(table)
        # table = np.array([((i / 255.0) ** invGamma) * 255
        #         for i in np.arange(0, 256)]).astype("uint8")

        img =  cv2.LUT(img,table)

        return img
    

    def blurr_subtract_preprocess_layer(self,sub):
#         sub = cv2.equalizeHist(sub)
        blurr = cv2.blur(sub.copy(),ksize=(self.k,self.k))
        img = sub.copy().astype(float) - blurr.astype(float) 
        return img - np.min(img)
    
    
    def local_binary_pattern(self,image, test=False):
    
        if len(image.shape[:])==3:
            print("converting image to grayscale")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)

        kernel = np.ones((3,3))

        def sliding_2d_window(img, kernel):

            m,n=img.shape[:2]
            j,k = kernel.shape[:2]
            pad = (k-1)//2
            out = np.ones((m,n),np.float64)*8

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

                    center_pixel = img[j,i]
                    pattern = img[xi:xf,yi:yf].copy() > center_pixel

                    out[j:j+1,i:i+1] = np.sum(pattern)

            return  out

        output = sliding_2d_window(image, kernel)

        if test:
            print("kernel")
            print(kernel)
            print("output image: top left")
            print(output[:3,:3])
            print("output image: bottom right")
            print(output[-3:,-3:])

        return output


    def get_peak_features(self,band):


        mat = np.zeros(band.shape)
        img = band[:,:].copy()
        blurr = cv2.blur(img.copy(),ksize=(self.k,self.k))

        z = np.zeros(img.shape[:])
        o = self.local_binary_pattern(blurr, test=False)
        msk = (o<self.peak_num) 
        z[msk] = img[msk]
        mat[:,:] = z
        return mat


# The CameraImages class takes a dictionary for each camera type and the processing functions
# The ProccesingFunctions class takes the parmaters for the processing functions
i=0
for key in phone_dict.keys():
    print("Camera phone: ",key)

    tmp  = ProcessingFunctions(k=3,peak_num=7)
    phone = CameraImages(phone_dict[key][:1000],tmp.get_peak_features,tmp.histogram_mask_mean_counts,tmp.histogram_mask_dark_pixels)
    phone.make_tiles()
    phone.category(lut[key])
    print(np.array(phone.ret_).shape[:])
    
    if i==0:
        images = phone.ret_
        categories = phone.cat_
        indx = phone.indx_
    else:
        images = np.vstack((images,phone.ret_))
        categories = np.concatenate((categories, phone.cat_))
        indx = np.concatenate((indx, phone.indx_))
    i+=1
    
# CNN implementation

import pandas as pd
from keras.utils import np_utils

target = np_utils.to_categorical(categories)
tr_mask = pd.Series(indx).isin(choose_img).values
v_mask = ~tr_mask

indx_tr = indx[tr_mask]
indx_v = indx[v_mask]
target_tr = target[tr_mask]
target_v = target[v_mask]
images_tr = images[tr_mask]
images_v = images[v_mask]

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Activation
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras import backend as K

def ConvBlock(model, layers, filters):
    '''Create [layers] layers consisting of zero padding, a convolution with [filters] 3x3 filters and batch normalization. Perform max pooling after the last layer.'''
    for i in range(layers):
#         model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters, (3, 3), activation='relu'))
        model.add(BatchNormalization(axis=3))
#     model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
def create_model():
    '''Create the FCN and return a keras model.'''

    model = Sequential()

    # Input image: 75x75x3
    model.add(Lambda(lambda x: x, input_shape=(100, 100, 3)))
    ConvBlock(model, 1, 32)
    # 37x37x32
    ConvBlock(model, 1, 32)
    # 18x18x64
    ConvBlock(model, 1, 64)
    # 9x9x128
    ConvBlock(model, 1, 64)
    # 4x4x128
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(10, (3, 3), activation='relu'))
#     model.add(Dense(10, activation='sigmoid'))


    model.add(GlobalAveragePooling2D())
    # 4x4x2
    model.add(Activation('softmax'))
    
#     model.add(Dense(128, 4))
#     model.add(ZeroPadding2D((1, 1)))
#     model.add(Conv2D(2, (3, 3), activation='relu'))
#     model.add(GlobalAveragePooling2D())
    # 4x4x2
    model.add(Activation('softmax'))
    
    return model

# Create the model and compile
model = create_model()
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
model.summary()

init_epo = 0
num_epo = 5
end_epo = init_epo + num_epo

print ('lr = {}'.format(K.get_value(model.optimizer.lr)))
history = model.fit(images_tr, target_tr, validation_data=(images_v, target_v),
                    batch_size=32,
                    epochs=end_epo,
                    initial_epoch=init_epo)
init_epo += num_epo
end_epo = init_epo + num_epo

tr_predict = model.predict(images_tr)
    
    
    
