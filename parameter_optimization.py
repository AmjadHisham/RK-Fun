class POptim:
    
    def __init__(self, LEN=1, SHAPE=1):

        self.shape_ = SHAPE
        self.len_ = LEN
        self.clear()

    def clear(self):

        self.w = np.zeros(self.shape_).reshape((self.shape_,1))
        self.wh_indx = np.arange(0,self.len_).reshape((self.len_,1))
        self.wh = np.zeros(self.shape_*self.len_).reshape((self.shape_,self.len_))
        
    def SetParam(self,WH):
        
        self.wh = WH
        self.len_ = wh.shape[1]
        self.shape_ = wh.shape[0]
        self.w = self.wh[:,-1].reshape((self.shape_,1))
        self.wh_indx = np.arange(0,self.len_).reshape((self.len_,1))
        
    def SetMetric(self,G):
        self.metric_ = G
        
        
import time

class ADADelta:
    """PID Controller
    """

    def __init__(self, W, PH=0.2, MH=0.2, A=1):

        self.w = W
        self.Kph = PH
        self.Kmh = MH
        self.Ka = A
        self.clear()
        
    def clear(self):

        self.vh = 0.00
        self.ph = 0.00
        self.update = 0.0
        self.rms_v = 0.0
        self.i = 0

    def update_(self, gradient):

        if self.i==0:
            v = gradient**2
            self.rms_v = np.sqrt(v+0.00000001)
            self.update= -1.0*(np.sqrt(0.00000001)/self.rms_v)*gradient
            self.ph = self.update**2

        else:

            v = (self.vh)*self.Kmh + (1-self.Kmh)*gradient**2
            
            self.rms_v = np.sqrt(v+0.00000001)
            self.update= -1.0*(np.sqrt(self.ph+0.00000001)/self.rms_v)*gradient
        
        self.i += 1
            
        self.ph = self.ph*self.Kph + (1-self.Kph)*self.update**2

        self.vh = v
        
        try:
            self.w = self.w + self.update*self.Ka
        except:
            print(self.w, self.update, self.Ka)


    def setKph(self, ph):
        """Determines how aggressively the PID reacts to the current error with setting Proportional Gain"""
        self.Kph = ph

    def setKmh(self, mh):
        """Determines how aggressively the PID reacts to the current error with setting Integral Gain"""
        self.Kmh = mh

    def setKa(self, a):
        """Determines how aggressively the PID reacts to the current error with setting Derivative Gain"""
        self.Ka = a
        
        
class PID:
    """PID Controller
    """

    def __init__(self, P=0.2, I=0.0, D=0.0):

        self.Kp = P
        self.Ki = I
        self.Kd = D

        self.sample_time = 0.00
        self.current_time = time.time()
        self.last_time = self.current_time

        self.clear()

    def clear(self):
        """Clears PID computations and coefficients"""
        self.SetPoint = 0.0

        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0

        # Windup Guard
        self.int_error = 0.0
        self.windup_guard = 20.0

        self.output = 0.0

    def update(self, feedback_value):
        """Calculates PID value for given reference feedback
        .. math::
            u(t) = K_p e(t) + K_i \int_{0}^{t} e(t)dt + K_d {de}/{dt}
        .. figure:: images/pid_1.png
           :align:   center
           Test PID with Kp=1.2, Ki=1, Kd=0.001 (test_pid.py)
        """
        error = self.SetPoint - feedback_value

        self.current_time = time.time()
        delta_time = self.current_time - self.last_time
        delta_error = error - self.last_error

        if (delta_time >= self.sample_time):
            self.PTerm = self.Kp * error
            self.ITerm += error * delta_time

            if (self.ITerm < -self.windup_guard):
                self.ITerm = -self.windup_guard
            elif (self.ITerm > self.windup_guard):
                self.ITerm = self.windup_guard

            self.DTerm = 0.0
            if delta_time > 0:
                self.DTerm = delta_error / delta_time

            # Remember last time and last error for next calculation
            self.last_time = self.current_time
            self.last_error = error

            self.output = self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)

    def setKp(self, proportional_gain):
        """Determines how aggressively the PID reacts to the current error with setting Proportional Gain"""
        self.Kp = proportional_gain

    def setKi(self, integral_gain):
        """Determines how aggressively the PID reacts to the current error with setting Integral Gain"""
        self.Ki = integral_gain

    def setKd(self, derivative_gain):
        """Determines how aggressively the PID reacts to the current error with setting Derivative Gain"""
        self.Kd = derivative_gain

    def setWindup(self, windup):
        """Integral windup, also known as integrator windup or reset windup,
        refers to the situation in a PID feedback controller where
        a large change in setpoint occurs (say a positive change)
        and the integral terms accumulates a significant error
        during the rise (windup), thus overshooting and continuing
        to increase as this accumulated error is unwound
        (offset by errors in the other direction).
        The specific problem is the excess overshooting.
        """
        self.windup_guard = windup

    def setSampleTime(self, sample_time):
        """PID that should be updated at a regular interval.
        Based on a pre-determined sampe time, the PID decides if it should compute or return immediately.
        """
        self.sample_time = sample_time
        
        

#Loss & gradient functions
def liklihood_loss(x,y,w):
    
    hypothesis = x.dot(w)
    hypothesis = 1/(1+np.exp(-1*hypothesis))

    loss = -(1-y).T.dot(np.log(1-hypothesis)) - y.T.dot(np.log(hypothesis))
    return loss.flatten()[0]

def gradient_func(x,y,w):
    hypothesis = x.dot(w)
    loss = hypothesis-y
    gradient = x.T.dot(loss)
    return gradient
    
def least_sqs_loss(x,y,w):
    hypothesis = x.dot(w)
    loss = hypothesis-y
    return np.sum(loss**2)
    
    
def adadelta_(INPUT,OUTPUT,WH=[], PH=0.2, MH=0.2, A=1, L=100):
    
    y=OUTPUT.values
    mean_pred=np.sum((y-np.mean(y))**2)
    m,n =np.shape(INPUT.values)
    x=np.ones((m,n+1))
    x[:,1:]=INPUT.values
    
    if len(WH)==0:
        parameters = POptim(10,x.shape[1])
    else:
        parameters = POptim(10,x.shape[1])
        parameters.SetParam(WH)
        
    w = parameters.w
    wh = parameters.wh
    wh_indx = parameters.wh_indx

    ada = ADADelta(W=w, PH=PH, MH=MH, A=A)
    END = L

    convh = 0
    update_log = []
    gradient_list = []
    rms_v_log = []
    w_log = []
    location_log = []
    
    feedback_list = []
    time_list = []
    setpoint_list = []

    for i in range(1, END):
        gradient = gradient_func(x,y,ada.w)
        ada.update_(gradient)

        conv = np.sqrt(np.sum(gradient**2)*2.0/(n+1)) 
        
        if i==1:
            conv_init = conv
            conv_init_mag = len(str(int(conv)))
            
        
        if (conv>convh*1.5) & (i>1):
            convh = conv
            wh = np.column_stack((wh,w))
            wh = wh[:,1:]
            return wh
        

        # Sample every 100 iterations
        if i % 100 == 0:
            wh = np.column_stack((wh,w))
            wh = wh[:,1:]

        w = ada.w
        convh = conv
        
    return wh


def pid_(INPUT,OUTPUT,wh, loss_func, P=0.1, I=0.1, D=0):
    
    y=OUTPUT.values
    mean_pred=np.sum((y-np.mean(y))**2)
    m,n =np.shape(INPUT.values)
    x=np.ones((m,n+1))
    x[:,1:]=INPUT.values
    
    predict_indx = 10
    
    parameters = POptim(1,x.shape[1])
    parameters.SetParam(wh)
        
    w = parameters.w
    wh = parameters.wh
    wh_indx = parameters.wh_indx
    wh_len = parameters.len_

    pid_list = []
    feedback_list = []
    rsq_list = []
    for i_ in range(wh.shape[0]):
        z = Regression.numpy_simple_regression(wh_indx,wh[i_])
        rsq = Regression.numpy_rsq(wh_indx,wh[i_],z)

        pid = PID(P,I,D)
        pid.SetPoint = wh[i_][-1]
        print("set point", pid.SetPoint)
        pid_list.append(pid)
        rsq_list.append(rsq)
        feedback_list.append(np.array([1,wh_len+predict_indx]).dot(z.T))
        print("feedback",feedback_list[i_])
        
    loss = loss_func(x,y,w)
    
    convh_ = loss
    
    w_tmp  = w.copy()
    print(w_tmp)
    
    for i_ in range(wh.shape[0]):
        for p in range(0,100):    
            pid_list[i_].update(feedback_list[i_])
            output = pid_list[i_].output
            feedback_list[i_] += output
#             print(output)
            w_tmp[i_] = feedback_list[i_]

            loss = loss_func(x,y,w_tmp)
        
            conv_ = loss

            if (conv_<convh_):
                print(p)
                print("old parameter",w)
                w = w_tmp
                print("new parameter",w)
                
                wh = np.column_stack((wh,w))
                wh = wh[:,1:]
                break
                
                
    """ Can be used to simultaneously shift all parameters
    for p in range(0,100):    
        for i_ in range(wh.shape[0]):
            pid_list[i_].update(feedback_list[i_])
            output = pid_list[i_].output
            feedback_list[i_] += output
            w_tmp[i_] = feedback_list[i_]

        loss = liklihood_loss(x,y,w_tmp)
        conv_ = loss

        if (conv_<convh_):
            print(conv_)
            print("old parameter",w)
            w = w_tmp
            print("new parameter",w)

            wh = np.column_stack((wh,w))
            wh = wh[:,1:]
            break
    """         

    return wh
    
# Test learning method - regression     
if __name__ == "__main__":
    X = train[['sqft_living','bedrooms','bathrooms','bed_bath_rooms','bedrooms_squared']]

    y = train[["price"]]
    wh = adadelta_(X,y, [], 0.3, 0.9, 10, L=500000)
    for x in range(wh.shape[0]):
        plt.plot(wh[x])
    plt.show()

    for _ in range(100): 
        
        wh = pid_(X,y,wh,least_sqs_loss,2,1,0.001)
        for x in range(wh.shape[0]):
            plt.plot(wh[x])
        plt.show()
        
        wh = pid_(X,y,wh,least_sqs_loss,2,1,0)
        for x in range(wh.shape[0]):
            plt.plot(wh[x])
        plt.show()
    
        
        wh = pid_(X,y,wh,least_sqs_loss,1,1,0)
        for x in range(wh.shape[0]):
            plt.plot(wh[x])
        plt.show()
        
        wh = pid_(X,y,wh,least_sqs_loss,0.2,0.5,0)
        for x in range(wh.shape[0]):
            plt.plot(wh[x])
        plt.show()
        
        wh = pid_(X,y,wh,least_sqs_loss,0.2,0.2,0)
        for x in range(wh.shape[0]):
            plt.plot(wh[x])
        plt.show()
    
    
    
    print(wh[:,-1].flatten())
    print(true_w)
    print(true_w-wh[:,-1].flatten())
    
#Compare loss
INPUT = train[['sqft_living','bedrooms','bathrooms','bed_bath_rooms','bedrooms_squared']]
OUTPUT = train[["price"]]
y=OUTPUT.values
mean_pred=np.sum((y-np.mean(y))**2)
m,n =np.shape(INPUT.values)
x=np.ones((m,n+1))
x[:,1:]=INPUT.values

parameters = POptim(wh.shape[1],wh.shape[0])
parameters.SetParam(wh)
w = parameters.w
hypothesis = x.dot(w)
loss = hypothesis-y
print(np.sum((loss)**2))

hypothesis = x.dot(np.array(true_w).reshape(len(INPUT.columns)+1,1))
loss = hypothesis-y
print(np.sum((loss)**2))

# Test learning method - logistic regression     

if __name__ == "__main__":
    
    X = train[['sqft_living','bathrooms']]
    y = pd.DataFrame((train[['bedrooms']].values > 3).astype(np.int))
    
    wh = adadelta_(X,y, [], 0.9, 0.9, 1, L=500000)
    for x in range(wh.shape[0]):
        plt.plot(wh[x])
    plt.show()

    for _ in range(2):
        wh = adadelta_(X,y, wh, 0.5, 0.5, 1, L=500000)
        for x in range(wh.shape[0]):
            plt.plot(wh[x])
        plt.show()    
    
    
    for _ in range(50): 
    
        wh = pid_(X,y,wh,liklihood_loss, 2,1,0.001)
        for x in range(wh.shape[0]):
            plt.plot(wh[x])
        plt.show()
        
        wh = pid_(X,y,wh,liklihood_loss, 2,1,0)
        for x in range(wh.shape[0]):
            plt.plot(wh[x])
        plt.show()
        
        for p in [0.5, 0.2]:
            for i in [0.5, 0.2]:
                wh = pid_(X,y,wh,liklihood_loss,p,i,0)
                
                for x in range(wh.shape[0]):
                    plt.plot(wh[x])
                plt.show()
                
    
    print(wh[:,-1].flatten())
    print(true_w)
    
INPUT = train[['sqft_living','bathrooms']]
OUTPUT = pd.DataFrame((train[['bedrooms']].values > 2).astype(np.int))
y=OUTPUT.values
mean_pred=np.sum((y-np.mean(y))**2)
m,n =np.shape(INPUT.values)
x=np.ones((m,n+1))
x[:,1:]=INPUT.values

parameters = POptim(wh.shape[1],wh.shape[0])
parameters.SetParam(wh)
w = parameters.w
hypothesis = x.dot(w)
hypothesis = 1/(1+np.exp(-1*hypothesis))
loss = hypothesis-y

loss = np.sum(np.sqrt(np.abs(hypothesis-y)**2))
print(loss)
hypothesis = x.dot(np.array(true_w).reshape(3,1))
hypothesis = 1/(1+np.exp(-1*hypothesis))
loss = hypothesis-y
loss = np.sum(np.sqrt(np.abs(hypothesis-y)**2))
print(loss)
print(true_w-wh[:,-1].flatten())

# Test non-linear implementation
import cv2
img = cv2.imread("image_dump/football.png",0)

img1 = img[:150,:200]
img2 = img[30:180,10:210]

def img_border(img,k):
    m,n = img.shape[:]
    pad = (k)//2
    out = np.zeros((m+2*pad,n+2*pad),np.uint8)
    out[pad:m+pad,pad:n+pad] = img.copy()
    print(out[pad:m+pad,pad:n+pad].shape[:],img.shape[:])
    return out
        

plt.imshow(img1)
plt.show()

plt.imshow(img2)
plt.show()

cross = np.fft.fftn(img1) * np.conjugate(np.fft.fftn(img2))

cc = np.absolute(np.fft.ifftn(cross/np.absolute(cross)))

i_,j_ = np.where(cc == np.max(cc))
i_ = i_[0]
j_ = j_[0]
print(i_,j_)


def similarity_metric(img1,img2,shifts):
        x_,y_ = shifts[0],shifts[1]
        rows,cols = img1.shape[:];
#         print(img1.shape)
        
        top_left_y = 0;
        top_left_x = 0;
        bottom_right_y = cols;
        bottom_right_x = rows;

        top_left_y_prime = 0;
        top_left_x_prime = 0;
        bottom_right_y_prime = cols;
        bottom_right_x_prime = rows;

        
        if(x_ >= 0):
            top_left_y = x_;
            bottom_right_y_prime = cols-x_;
        else:
            bottom_right_y = cols+x_;
            top_left_y_prime = -1*x_;
        
        if(y_ >= 0):
            top_left_x = y_;
            bottom_right_x_prime = rows-y_;
        else:
            top_left_x_prime = -1*y_;
            bottom_right_x = rows+y_;
        
        rout = np.zeros(img1.shape)
        
        cout = np.zeros(img1.shape)
        
#         print(top_left_x,bottom_right_x)

        top_left_x_prime = np.int(top_left_x_prime)
        top_left_y_prime = np.int(top_left_y_prime)
        top_left_y = np.int(top_left_y)
        top_left_x = np.int(top_left_x)
        bottom_right_x_prime = np.int(bottom_right_x_prime)
        bottom_right_y_prime = np.int(bottom_right_y_prime)
        bottom_right_y = np.int(bottom_right_y)
        bottom_right_x = np.int(bottom_right_x)
        
        
        rout[top_left_x_prime:(bottom_right_x_prime),top_left_y_prime:(bottom_right_y_prime)] = img1[top_left_x:(bottom_right_x),top_left_y:(bottom_right_y)];

        cout[top_left_x_prime:(bottom_right_x_prime),top_left_y_prime:(bottom_right_y_prime)] = img2[top_left_x_prime:(bottom_right_x_prime),top_left_y_prime:(bottom_right_y_prime)];

    

        
        row_num = rout.size;

        Y = cout.reshape((1, row_num));

        rout = rout.reshape((1, row_num));
        offset = np.ones((1,row_num));

        X = np.vstack([offset,rout]); 
        
        
        X  = X.astype(np.float)
        Y  = Y.astype(np.float)
        
        A = X.dot(X.T)
        b = X.dot(Y.T)

        w_ = np.linalg.solve(A,b); 
        
        sst = np.sum((Y - np.mean(Y))**2);
        sse = np.sum((Y - w_.T.dot(X))**2);
        
        
#         plt.imshow(Y.reshape(img1.shape) - w_.T.dot(X).reshape(img1.shape))
#         plt.show()
        
#         print(sse,sst)
        rsq = 1 - sse/sst;
        return sse,rsq

def shift_matrices(mat1,mat2,phase_correlation=True,shifts=[]):
    
    if phase_correlation == True:
        cross = np.fft.fftn(mat1) * np.conjugate(np.fft.fftn(mat2))

        cc = np.absolute(np.fft.ifftn(cross/np.absolute(cross)))

        shifts = np.where(cc == np.max(cc))
    shifts = [x[0] for x in shifts]

    mat1 = np.squeeze(mat1)
    mat2 = np.squeeze(mat2)
    print("shifts:",shifts)
    for i in range(mat1.ndim):

        shift = shifts[i]
    #     shift*=-1
        shape_ = mat1.shape
        shape_ = list(shape_)
        
        if shift>mat1.shape[i]//2:
            shift = -1*(mat1.shape[i]-shift)
            print("shift greater than 1/2 mat len, reversing shift:",shift)        
        shape_[i] = abs(shift)
        
        print("updated shape:",shape_)
        f = np.zeros(tuple(shape_),np.uint8)


        if shift == 0:
            print("no shift detected")
            break
        
        if shift>0:
            tmp = np.delete(mat1,np.s_[:shift],axis=i)

            mat1 = np.concatenate((tmp,f), axis=i)

            tmp = np.delete(mat2,np.s_[-1*shift:],axis=i)
            mat2 = np.concatenate((tmp,f), axis=i)

            plt.imshow(mat1)
            plt.show()
            
            plt.imshow(mat2)
            plt.show()


        else:
            tmp = np.delete(mat1,np.s_[shift:],axis=i)

            mat1 = np.concatenate((f,tmp), axis=i)

            tmp = np.delete(mat2,np.s_[:-1*shift],axis=i)
            mat2 = np.concatenate((f,tmp), axis=i)
            
            
            plt.imshow(mat1)
            plt.show()
            plt.imshow(mat2)
            plt.show()
            

    return mat1,mat2

shifts = [[30],[10]]
shift_matrices(img1,img2,False,shifts)

def pid_imageshift(INPUT,OUTPUT,WH, P=0.1, I=0.1, D=0):
    
    img1 = INPUT
    img2 = OUTPUT
    rows,cols = img1.shape
    
    predict_indx = 10
    
    parameters = POptim(WH.shape[1],WH.shape[0])
    parameters.SetParam(WH)
    
    
    w = parameters.w
    wh = parameters.wh
    wh_indx = parameters.wh_indx
    wh_len = parameters.len_

    pid_list = []
    feedback_list = []
    rsq_list = []
    for i_ in range(wh.shape[0]):
        z = Regression.numpy_simple_regression(wh_indx,wh[i_])
        rsq = Regression.numpy_rsq(wh_indx,wh[i_],z)

        pid = PID(P,I,D)
        pid.SetPoint = wh[i_][-1]
        print("set point", pid.SetPoint)
        pid_list.append(pid)
        rsq_list.append(rsq)
        
        predict = np.array([1,wh_len+predict_indx]).dot(z.T)
        
        feedback_list.append(np.array(predict))
        
#         loss, rsq = similarity_metric(img1,img2,w_tmp)
        print("feedback",feedback_list[i_])
#         print('z',z,wh_len)

        
    y_,x_=feedback_list[0],feedback_list[1]

    if x_>cols-1:
        x_ = cols-1

    if y_>rows-1:
        y_ = rows-1

    if x_<-(cols-1):
        x_ = -(cols-1)

    if y_<-(rows-1):
        y_ = -(rows-1)

    feedback_list[0] = y_
    feedback_list[1] = x_
        
    loss,rsq = similarity_metric(img1,img2,w)
    convh_ = loss
    
    w_tmp  = w.copy()
#     print(w_tmp)
        
    for i_ in range(wh.shape[0]):
        for p in range(0,100):
    #             if rsq_list[i_]>0.9:
            pid_list[i_].update(feedback_list[i_])
            output = pid_list[i_].output
            feedback_list[i_] += output
#             print(output)
            w_tmp[i_] = np.ceil(feedback_list[i_])
        

#         print(w_tmp)

            loss, rsq = similarity_metric(img1,img2,w_tmp)
            conv_ = loss

            if (conv_<convh_):
                print(p)
                print("old parameter",w)
                w = w_tmp
                print("new parameter",w)
                
                wh = np.column_stack((wh,w))
                wh = wh[:,1:]
                break


    return wh

# Initialize random weights
heap = []
weights = np.random.randint(low=-20,high=0,size=(2,40))
for i in range(weights.shape[1]):
    x_,y_ = weights[:,i]
    sse,rsq = similarity_metric(img2,img1,(y_,x_))
    heap.append((sse,y_,x_))
sorted_heap = Regression.heapsort(heap)
sorted_heap
    
wh = np.array([[y_,x_] for _,y_,x_ in reversed(sorted_heap[:5]) ]).T
wh.shape

for _ in range(50):
    wh = pid_imageshift(img2,img1,wh,P=2, I=1, D=0)
    wh = pid_imageshift(img2,img1,wh,P=1, I=1, D=0)
    wh = pid_imageshift(img2,img1,wh,P=0.5, I=1, D=0)

print(wh)

import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline

def test_pid(P = 0.2,  I = 0.0, D= 0.0, L=100):
    """Self-test PID class
    .. note::
        ...
        for i in range(1, END):
            pid.update(feedback)
            output = pid.output
            if pid.SetPoint > 0:
                feedback += (output - (1/i))
            if i>9:
                pid.SetPoint = 1
            time.sleep(0.02)
        ---
    """
    pid = PID(P, I, D)

    pid.SetPoint=0.0
    pid.setSampleTime(0.01)

    END = L
    feedback = 0

    feedback_list = []
    time_list = []
    setpoint_list = []


    
    for i in range(1, END):
        pid.update(feedback)
        output = pid.output
        if pid.SetPoint > 0:
            feedback += output
        if i>9:
            pid.SetPoint = 1
        time.sleep(0.02)

        feedback_list.append(feedback)
        setpoint_list.append(pid.SetPoint)
        time_list.append(i)

    time_sm = np.array(time_list)
    time_smooth = np.linspace(time_sm.min(), time_sm.max(), 300)
    feedback_smooth = spline(time_list, feedback_list, time_smooth)

    plt.plot(time_smooth, feedback_smooth)
    plt.plot(time_list, setpoint_list)

    plt.xlabel('time (s)')
    plt.ylabel('PID (PV)')
    plt.title('TEST PID')


    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    test_pid(2, 1, 0.001, L=50)
    test_pid(2, 1, 0, L=50)
    test_pid(1, 1, 0, L=50)
    test_pid(0.5, 0.2,0, L=50)
    test_pid(0.2, 0.5,0, L=50)
    test_pid(0.2, 0.2,0, L=50)