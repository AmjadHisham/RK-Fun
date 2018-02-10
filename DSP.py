import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import heapq

def heapsort(iterable):
    h = []
    for value in iterable:
        heapq.heappush(h, value)
    return [heapq.heappop(h) for i in range(len(h))]


def filter_freq_domain(signal,center,band,threshold):
    filtered=[]
    N = len(signal)*2
    for f,x in signal:
        if np.absolute(f)>center+band or np.absolute(f)<center-band or (2.0/N *np.abs(x))<=threshold:
            pass
        else:
            filtered.append((f,x))
    return np.array(filtered)


def datetime_index(x):
    z  = np.ones(x.shape[:])
    for t in range(len(x)):
        if t>=1:
            z[t]=(x[t]-x[t-1]).days+z[t-1]
    return z-1   

def time_diff_variable(x,lag):
    z = np.zeros(x.shape[:])
    for t in range(len(x)):
        if t>=lag:
            z[t]=x[t]-x[t-lag]
    return z[lag:]


def get_frequency_domain(signal):
    N = len(signal)
    yf = np.fft.fft(signal)
    freq= np.fft.fftfreq(len(yf))
    return np.column_stack((freq[:N//2],yf[:N//2]))  



def generate_sin_waves(index,period):
    seasonal_x = np.zeros((len(index),1))
    for i in range(int(period)):
        index_i = index+i
        x = np.array([x%period for x in index_i]).reshape(len(index),1)
        frequency_components = 2*np.pi*np.array([1./period]).reshape((1,1))
        trig_args = np.dot(frequency_components,x.T ).T
        seasonal_x = np.column_stack((seasonal_x,np.cos(trig_args)))
    return seasonal_x


def fit_sin_waves(index,target,period):
    
    x = np.array([x%period for x in index]).reshape(len(index),1)
    frequency_components = 2*np.pi*np.array([1./period]).reshape((1,1))
    trig_args = np.dot(frequency_components,x.T ).T
    seasonal_x = np.sin(trig_args)
    cross = np.fft.fft(seasonal_x.flatten())*np.conjugate(np.fft.fft(target.flatten()))
    #cross = (cross - np.mean(cross))/np.std(cross)
    cc = np.absolute(np.fft.ifft(cross))
    #print(np.max(cc))
    #print(cc)
    #plt.plot(cc)
    #plt.show()
    
    half =  round(len(index)/2+0.5,0)
    g = np.concatenate([np.arange(0,half),np.arange(half,len(index))-len(index)])
    shift = [(val,i) for i,val in zip(g,cc.flatten()) if val==np.max(cc.flatten())]
    sorted_shift = heapsort(shift)[:]

    seasonal_x = np.zeros((len(index),1))
    shift_list = []
    for val,t in sorted_shift:
        if t not in shift_list:
            print(t)
            x = np.array([(x+t)%period for x in index]).reshape(len(index),1)
            frequency_components = 2*np.pi*np.array([1./period]).reshape((1,1))
            trig_args = np.dot(frequency_components,x.T ).T
            seasonal_x = np.column_stack((seasonal_x,np.sin(trig_args)))
            shift_list.append(t)

    return seasonal_x[:,1:],shift_list

def generate_fit_sin_waves(index,period,shift):
    
    seasonal_x = np.zeros((len(index),1))
    for t in shift:
        x = np.array([(x+t)%period for x in index]).reshape(len(index),1)
        frequency_components = 2*np.pi*np.array([1./period]).reshape((1,1))
        trig_args = np.dot(frequency_components,x.T ).T
        seasonal_x = np.column_stack((seasonal_x,np.sin(trig_args)))

    return seasonal_x[:,1:]

def generate_sawtooth_waves(index,period):
    seasonal_x = np.zeros((len(index),1))
    for i in range(int(period)):
        index_i = index+i
        seq = np.array([x%period for x in index_i])
        seasonal_x = np.column_stack((seasonal_x,seq))
    return seasonal_x[:,1:]

def fit_sawtooth_waves(index,period):
    seasonal_x = np.zeros((len(index),1))
    x = np.array([x%period for x in index_i])

    return seasonal_x[:,1:]

def generate_reverse_sawtooth_waves(index,period):
    seasonal_x = np.zeros((len(index),1))
    for i in range(int(period)):
        index_i = index+i
        seq = np.array([period-x%period for x in index_i])
        seasonal_x = np.column_stack((seasonal_x,seq))
    return seasonal_x[:,1:]

def fit_impulse_waves(index,target,period):
    
    def square_func(t,period):
        if t%period == 0:
            return 1
        else:
            return 0

    x = np.array([square_func(x,period) for x in index]).reshape(len(index),1)
    
    cross = np.fft.fft(x.flatten())*np.conjugate(np.fft.fft(target.flatten()))
    #cross = (cross - np.mean(cross))/np.std(cross)
    cc = np.absolute(np.fft.ifft(cross))
    #print(np.max(cc))
    #print(cc)
    #plt.plot(cc)
    #plt.show()
    
    half =  round(len(index)/2+0.5,0)
    g = np.concatenate([np.arange(0,half),np.arange(half,len(index))-len(index)])
    shift = [(val,i) for i,val in zip(g,cc.flatten()) if val==np.max(cc.flatten())]
    sorted_shift = heapsort(shift)[:]

    seasonal_x = np.zeros((len(index),1))
    shift_list = []
    for val,t in sorted_shift:
        if t not in shift_list:
            x = np.array([square_func(x+t,period) for x in index]).reshape(len(index),1)
            seasonal_x = np.column_stack((seasonal_x,x))
            shift_list.append(t)

    return seasonal_x[:,1:],shift_list

def generate_fit_impulse_waves(index,period,shift):
    def square_func(t,period):
        if t%period == 0:
            return 1
        else:
            return 0
    
    seasonal_x = np.zeros((len(index),1))
    for t in shift:

        x = np.array([square_func(x+t,period) for x in index]).reshape(len(index),1)
        seasonal_x = np.column_stack((seasonal_x,x))

    return seasonal_x[:,1:]

def generate_impluse_waves(index,period):
    def square_func(t,period):
        if t%period == 0:
            return 1
        else:
            return -1
    seasonal_x = np.zeros((len(index),1))
    for i in range(int(period)):
        index_i = index+i
        seq = np.array([square_func(x,period) for x in index_i])
        seasonal_x = np.column_stack((seasonal_x,seq))
    return seasonal_x[:,1:]


def generate_neg_impluse_waves(index,period):
    def square_func(t,period):
        if t%period == 0:
            return -1
        else:
            return 1
    seasonal_x = np.zeros((len(index),1))
    for i in range(int(period)):
        index_i = index+i
        seq = np.array([square_func(x,period) for x in index_i])
        seasonal_x = np.column_stack((seasonal_x,seq))
    return seasonal_x[:,1:]

def generate_square_waves(index,period):
    def square_func(t,period):
        if t%period <period/2.:
            return 1
        else:
            return -1
    seasonal_x = np.zeros((len(index),1))
    for i in range(int(period)):
        index_i = index+i
        seq = np.array([square_func(x,period) for x in index_i])
        seasonal_x = np.column_stack((seasonal_x,seq))
    return seasonal_x[:,1:]


def get_sequences(index,period, plot=False):
    N = len(index)
    seasonal_list=[]
    #index = np.arange(start_index,len(df)+start_index)
        
        #df['seq_'+str(period)] = np.linspace(start_index//period+1/period,len(df)//period,len(df))
    seq = [x%period for x in index]

    if plot:
        unfiltered  = get_frequency_domain(seq)
        y_abs=( 2.0/N * np.abs(unfiltered[1:,1]))
        plt.plot(unfiltered[1:,0],y_abs)
        plt.xlabel("frequency")
        plt.ylabel("absolute magnitude")
        plt.title("Frequency domain, Sequence period = {}".format((period)))
        plt.show()
    return seq

def get_sequence_freq(period,var,std_thresh, plot=False):
    
    
    N=len(var.values.flatten())
    start_index=1
    index = np.arange(start_index,N+start_index)
    unfiltered  = get_frequency_domain( [x%period for x in index])
    #unfiltered  = get_frequency_domain(var.values.flatten())
  
    y_abs=( 2.0/N * np.abs(unfiltered[1:,1]))
    threshold = np.mean(y_abs)+std_thresh*np.std(y_abs)
    
    print('threshold',threshold)
    print('frequencies ',np.absolute(filter_freq_domain(unfiltered, 1,1,threshold)))
    
    if plot:
        plt.plot(unfiltered[1:,0],y_abs)
        plt.xlabel("frequency")
        plt.ylabel("absolute magnitude")
        plt.title(str(list(var.columns)[0])+": frequency domain")
        plt.show()
    
    component_list=np.absolute(filter_freq_domain(unfiltered,1,1,threshold=threshold))
    frequency_list = set([1/round(1/f2,1) for (f2,h2) in component_list if f2>0])
    
    return list(frequency_list)


def get_seasonal_predictors(var,z_list):
    m,n =np.shape(var.values)
    x=var.values
    print("frequencies used in ", list(var.columns)[0]," : ",z_list)
    frequency_components=np.array(z_list)
    frequency_components = 2*np.pi*np.outer(frequency_components,np.ones(x.shape[1])).reshape((len(frequency_components),x.shape[1]))

    trig_args = np.dot(frequency_components,x.T ).T
    seasonal_x = np.column_stack((np.sin(trig_args),np.cos(trig_args)))
    return seasonal_x