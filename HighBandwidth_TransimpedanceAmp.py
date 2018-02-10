import numpy as np
import matplotlib.ticker as plticker
import matplotlib.pyplot as plt
import random
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.io import loadmat
from scipy.io import savemat
import numpy as np
from scipy.stats import norm
import matplotlib.patches as mpatches

##
simulation_num = 1
snr_sync = []
snr_unsync = []
for _ in range(simulation_num):

    line_number = 15
    plt_laser = []
    plt_pulse = []
    plt_highbw_transimpndance = []
    plt_sync_digitizer = []
    plt_unsync_digitizer = []
    plt_peaks = []
    for line in range(line_number):

        beam_num = 20
        sample_f = 1/1e-11
        t_norm = np.arange(0,12.5e-9,1/sample_f)
        t = t_norm
        t_plot = np.arange(0,12.5e-9*beam_num*1e9,1e9/sample_f)

        mat = np.zeros((beam_num,len(t)))
        peaks = np.zeros((beam_num,len(t)))

        for n in range(beam_num):

            for i in range(np.random.poisson(1)):
                time = np.random.exponential(2.5e-9)
                diff = abs(t - time)
                indx = np.where(diff == np.min(diff))[0][0]


                if indx > 50 and indx<mat.shape[1]-50:
                    peaks[n,indx] += 250*np.max(norm.pdf(np.linspace(norm.ppf(0.001),norm.ppf(0.999), 100)))
                    mat[n,indx-50:indx+50] += 250*norm.pdf(np.linspace(norm.ppf(0.001),norm.ppf(0.999), 100))


        plt_peaks.append(peaks)
        from scipy import signal

        b, a = signal.bessel(7, 16.1e8, 'low', analog=True)
        w, h = signal.freqs(b, a, worN=np.arange(0,sample_f,sample_f/(mat.size)))


        A = 0.7
        R = 500
        Ci = 0
        Cf = 0
        s = 1j*np.arange(0,sample_f,sample_f/(mat.size))
        w0 = 80e6

        Tf = -A*(w0/(s+w0))*R/(s*Ci*R+(1+A*(w0/(s+w0)))*(1+s*Cf*R))
        highbw_transimped_amp = np.real(np.fft.ifft(np.fft.fft(mat.flatten().copy())*Tf))

        xmarks=np.array([i for i in range(0,mat.size,len(t))])
        haha = np.zeros(mat.size)
        for x in xmarks:
            haha[x]=np.max(mat)


        digitizer_sample_highbw= np.zeros(mat.size)
        digitizer_sample_highbw_unsynced= np.zeros(mat.size)

        time = 2.5e-9
        diff = abs(t - time)
        indx = np.where(diff == np.min(diff))[0]


        digitizer_sample_highbw[:int(indx)]=highbw_transimped_amp[0]
        digitizer_sample_highbw_unsynced[:int(indx)]=highbw_transimped_amp[0]

        for i in np.arange(indx,mat.size,len(t_norm)):
            i = i + 34
            digitizer_sample_highbw[int(i):int(i)+len(t)]=highbw_transimped_amp[int(i)]


        for i in np.arange(0,mat.size,len(t_norm)):

            i = i + np.random.randint(0,1000)
            digitizer_sample_highbw_unsynced[int(i):int(i)+len(t)]=highbw_transimped_amp[int(i)]


        plt_pulse.append(mat.flatten())
        plt_laser.append(haha)
        plt_sync_digitizer.append(digitizer_sample_highbw)
        plt_unsync_digitizer.append(digitizer_sample_highbw_unsynced)
        plt_highbw_transimpndance.append(highbw_transimped_amp)



fig = plt.figure(figsize=(15,30))
# fig.suptitle("Laser pulses & Possion Current Pulses", fontsize=26)
for i in range(line_number):
    ax = plt.subplot(line_number,1,i+1)
    ax.plot(t_plot,plt_pulse[i],'b')
    ax.plot(t_plot,plt_laser[i],'r')
    ax.set_ylabel("mA")
    
plt.xlabel("nano seconds")
plt.show()

fig = plt.figure(figsize=(15,30))
# fig.suptitle("Transimpedenance Amplifier - High Cut-Off Frequency", fontsize=26)
for i in range(line_number):
    ax = plt.subplot(line_number,1,i+1)
    ax.plot(t_plot,plt_highbw_transimpndance[i],'b')
    ax.set_ylabel("mV")
plt.xlabel("nano seconds")
plt.show()




    
#     savemat("Matrix_Values_Impulses/mat_"+str(xxx),{"impulses":mat.flatten(),"highbw_transimped_amp":highbw_transimped_amp,"digitizer_sample_highbw":digitizer_sample_highbw,"lowbw_transimped_amp":lowbw_transimped_amp,"digitizer_sample_lowbw":digitizer_sample_lowbw,"beams":haha})

##

import cv2

x = np.array(plt_sync_digitizer)
x = np.sum(x.reshape(line_number,beam_num,len(t)),axis=2) * -1
x  = (x - np.mean(x))/np.std(x)
synced_responses = x - np.min(x)
# x = cv2.resize(x,(250,250))
ax = plt.figure(figsize=(15,15))
# ax = plt.subplot(line_number,1,line+1)
plt.imshow(synced_responses,cmap='bone')
plt.show()

x = np.array(plt_unsync_digitizer)
x = np.sum(x.reshape(line_number,beam_num,len(t)),axis=2) * -1
x  = (x - np.mean(x))/np.std(x)
unsynced_responses  = x - np.min(x)
# x = cv2.resize(x,(250,250))
ax = plt.figure(figsize=(15,15))
# ax = plt.subplot(line_number,1,1)
plt.imshow(unsynced_responses,cmap='bone')
plt.show()

x = np.array(plt_peaks)
x = x
# x = (x - np.min(x))/(np.max(x)-np.min(x)) * 255
actual_response = np.sum(x.reshape(line_number,beam_num,len(t)),axis=2)
# x = cv2.resize(x,(250,250))
ax = plt.figure(figsize=(15,15))
# ax = plt.subplot(line_number,1,1)
plt.imshow(actual_response,cmap='bone')
plt.show()



n = len(unsynced_responses.flatten())
X = unsynced_responses.astype(np.int16).reshape(n,1)

Y = actual_response.astype(np.int16).flatten()

A=X.T.dot(X)
b=X.T.dot(Y)
z = np.linalg.solve(A,b)
errors = Y-X.dot(z)

SSE = np.sum((Y-X.dot(z))**2)

SST = np.sum((actual_response.astype(np.int16)-np.mean(actual_response.astype(np.int16)))**2)
# z,1-SSE/SST,SSE

print("mean error: ", np.mean(errors), "variance of errors: ", np.var(errors), "sum of squared error: ", SSE)

SNR = np.var(actual_response)/np.var(errors)
SNR

n = len(synced_responses.flatten())
X = synced_responses.astype(np.int16).reshape(n,1)

Y = actual_response.astype(np.int16).flatten()

A=X.T.dot(X)
b=X.T.dot(Y)
z = np.linalg.solve(A,b)

# z = Regression.numpy_simple_regression(X,Y)
# SSE = Regression.numpy_SSE(X,Y,z)

SSE = np.sum((Y-X.dot(z))**2)

SST = np.sum((actual_response.astype(np.int16)-np.mean(actual_response.astype(np.int16)))**2)
errors = Y-X.dot(z)
print("mean error: ", np.mean(abs(errors)), "variance of errors: ", np.var(errors), "sum of squared error: ", SSE)
# z,1-SSE/SST,SSE

SNR = np.var(actual_response)/np.var(errors)
SNR

fig = plt.figure(figsize=(15,15))
# fig.suptitle("Laser pulses & Possion Current Pulses", fontsize=26)
for i in range(1):
    ax = plt.subplot(4,1,1)
    ax.plot(t_plot[:15000],plt_pulse[i][:15000],'b')
    ax.plot(t_plot[:15000],plt_laser[i][:15000],'r')
    ax.set_title("PMT Output")
    ax.set_ylabel("mA")

# plt.xlabel("nano seconds")
# plt.show()

# fig = plt.figure(figsize=(15,15))
# fig.suptitle("Transimpedenance Amplifier - High Cut-Off Frequency", fontsize=26)

for i in range(1):
    ax = plt.subplot(4,1,2)
    green_patch = mpatches.Patch(color='tab:cyan', label='Sychronized Digitizer')
    plt.legend(handles=[green_patch])
    ax.plot(plt_sync_sample_loc[i][:15000],'tab:cyan')
    ax.plot(plt_highbw_transimpndance[i][:15000],'b')
    ax.set_title("Transimpedance Amplifier Output")
    ax.set_ylabel("mV")

    
for i in range(1):
    ax = plt.subplot(4,1,3)
    red_patch = mpatches.Patch(color='tab:orange', label='Unsychronized Digitizer')
    plt.legend(handles=[red_patch])
    ax.plot(plt_unsync_sample_loc[i][:15000],'tab:orange')
    ax.plot(plt_highbw_transimpndance[i][:15000],'b')
    ax.set_title("Transimpedance Amplifier Output")
    ax.set_ylabel("mV")
    
plt.xlabel("nano seconds")
plt.show()



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.io import loadmat
from scipy.io import savemat
import pyqtgraph as pg
import numpy as np
from scipy.stats import norm
import matplotlib.patches as mpatches


simulation_num = 5000
snr_sync = []
snr_unsync = []
for _ in range(simulation_num):

    line_number = 15
    plt_laser = []
    plt_pulse = []
    plt_highbw_transimpndance = []
    plt_sync_digitizer = []
    plt_unsync_digitizer = []
    plt_peaks = []
    for line in range(line_number):

        beam_num = 20
        sample_f = 1/1e-11
        t_norm = np.arange(0,12.5e-9,1/sample_f)
    #     t = np.arange(0,100e-9,1/sample_f)
        t = t_norm
        t_plot = np.arange(0,12.5e-9*beam_num*1e9,1e9/sample_f)
    #     t_plot = np.arange(0,100e-9*beam_num*1e9,1e9/sample_f)
        mat = np.zeros((beam_num,len(t)))
        peaks = np.zeros((beam_num,len(t)))

        for n in range(beam_num):

            for i in range(np.random.poisson(1)):
                time = np.random.exponential(2.5e-9)
                diff = abs(t - time)
                indx = np.where(diff == np.min(diff))[0][0]

#                 print(i,time,indx)
                if indx > 50 and indx<mat.shape[1]-50:

                    peaks[n,indx] += 250*np.max(norm.pdf(np.linspace(norm.ppf(0.001),norm.ppf(0.999), 100)))
                    mat[n,indx-50:indx+50] += 250*norm.pdf(np.linspace(norm.ppf(0.001),norm.ppf(0.999), 100))

        plt_peaks.append(peaks)
        from scipy import signal

        b, a = signal.bessel(7, 16.1e8, 'low', analog=True)
        w, h = signal.freqs(b, a, worN=np.arange(0,sample_f,sample_f/(mat.size)))


        A = 0.7
        R = 500
        Ci = 0
        Cf = 0
        s = 1j*np.arange(0,sample_f,sample_f/(mat.size))

        w0 = 2500e3
        Tf = -A*(w0/(s+w0))*R/(s*Ci*R+(1+A*(w0/(s+w0)))*(1+s*Cf*R))

        lowbw_transimped_amp = np.real(np.fft.ifft(np.fft.fft(mat.flatten().copy())*Tf))


        lowbw_bessel = np.real(np.fft.ifft(np.fft.fft(lowbw_transimped_amp)*h))


    #     A = 0.007
        Ci = 0
        Cf = 0
        w0 = 80e6

        b, a = signal.bessel(7, [2*np.pi*20e3,2*np.pi*16.1e6], 'bandpass', analog=True)
        w, h = signal.freqs(b, a, worN=np.arange(0,sample_f,sample_f/(mat.size)))

        s = 1j*np.arange(0,sample_f,sample_f/(len(mat.flatten())))
        Tf = -A*(w0/(s+w0))*R/(s*Ci*R+(1+A*(w0/(s+w0)))*(1+s*Cf*R))
        highbw_transimped_amp = np.real(np.fft.ifft(np.fft.fft(mat.flatten().copy())*Tf))
        highbw_bessel = np.real(np.fft.ifft(np.fft.fft(highbw_transimped_amp)*h))


        xmarks=np.array([i for i in range(0,mat.size,len(t))])
        haha = np.zeros(mat.size)
        for x in xmarks:
            haha[x]=np.max(mat)


        digitizer_sample_highbw= np.zeros(mat.size)

        digitizer_sample_highbw_unsynced= np.zeros(mat.size)


        time = 2.5e-9
        diff = abs(t - time)
        indx = np.where(diff == np.min(diff))[0]


        digitizer_sample_highbw[:int(indx)]=highbw_transimped_amp[0]
        digitizer_sample_highbw_unsynced[int(i):int(i)+len(t)]=highbw_transimped_amp[0]

        for i in np.arange(indx,mat.size,len(t_norm)):

            i = i + 34

            digitizer_sample_highbw[int(i):int(i)+len(t)]=highbw_transimped_amp[int(i)]
            


        for i in np.arange(0,mat.size,len(t_norm)):

            i = i + np.random.randint(0,1000)
            
            digitizer_sample_highbw_unsynced[int(i):int(i)+len(t)]=highbw_transimped_amp[int(i)]


        plt_pulse.append(mat.flatten())
        plt_laser.append(haha)
        plt_sync_digitizer.append(digitizer_sample_highbw)
        plt_unsync_digitizer.append(digitizer_sample_highbw_unsynced)
        plt_highbw_transimpndance.append(highbw_transimped_amp)



    x = np.array(plt_sync_digitizer)
    x = np.sum(x.reshape(line_number,beam_num,len(t)),axis=2) * -1
    x  = (x - np.mean(x))/np.std(x)
    synced_responses = x - np.min(x)


    x = np.array(plt_unsync_digitizer)
    x = np.sum(x.reshape(line_number,beam_num,len(t)),axis=2) * -1
    x  = (x - np.mean(x))/np.std(x)
    unsynced_responses  = x - np.min(x)

    x = np.array(plt_peaks)
    x = x
    actual_response = np.sum(x.reshape(line_number,beam_num,len(t)),axis=2)



    n = len(unsynced_responses.flatten())
    X = unsynced_responses.astype(np.int16).reshape(n,1)

    Y = actual_response.astype(np.int16).flatten()

    A=X.T.dot(X)
    b=X.T.dot(Y)
    z = np.linalg.solve(A,b)
    errors_unsync = Y-X.dot(z)
    
    n = len(synced_responses.flatten())
    X = synced_responses.astype(np.int16).reshape(n,1)

    Y = actual_response.astype(np.int16).flatten()

    A=X.T.dot(X)
    b=X.T.dot(Y)
    z = np.linalg.solve(A,b)
    errors_sync = Y-X.dot(z)
    
    snr_sync.append(np.var(actual_response)/np.var(errors_sync))
    snr_unsync.append(np.var(actual_response)/np.var(errors_unsync))
    
    
    
#     savemat("Matrix_Values_Impulses/mat_"+str(xxx),{"impulses":mat.flatten(),"highbw_transimped_amp":highbw_transimped_amp,"digitizer_sample_highbw":digitizer_sample_highbw,"lowbw_transimped_amp":lowbw_transimped_amp,"digitizer_sample_lowbw":digitizer_sample_lowbw,"beams":haha})

np.mean(snr_sync),np.mean(snr_unsync)

fig = plt.figure(figsize=(15,10))

df = pd.read_csv("Workbook1.csv")

plt.plot(df.ns,df.V,'-')

plt.xlabel("ns")
plt.ylabel("mV")
plt.xlim([0,73])
plt.show()

# fig = plt.figure(figsize=(15,15))

shifts = []
avg_cc = []
# fig.suptitle("Laser pulses & Possion Current Pulses", fontsize=26)
N = len(t_plot)    
freq= np.fft.fftfreq(N)
g = np.concatenate([np.arange(0,half),np.arange(half,N)-N])
for i in range(line_number):

    cross = np.fft.fft(plt_pulse[i])*np.conjugate(np.fft.fft(plt_highbw_transimpndance[i]))
    cc = np.absolute(np.fft.ifft(cross))
    avg_cc.append(cc)
    
    half =  round(N/2+0.5,0)
    shift = [(val,int(i)) for i,val in zip(g,cc.flatten()) if val==np.max(cc.flatten())]
    print(shift)
    
    plt.plot(g,cc.flatten())

plt.show()
avg_cc = np.array(avg_cc)
print(freq.shape[:])

avg_cc = np.sum(avg_cc,axis=0)


half =  round(N/2+0.5,0)

shift = [(val,int(i)) for i,val in zip(g,avg_cc.flatten()) if val==np.max(avg_cc.flatten())]
print(shift)
shifts.append(shift[0][1])
    
np.mean(shifts)
#     ax = plt.subplot(4,1,1)
#     ax.plot(t_plot[:15000],plt_pulse[i][:15000],'b')
#     ax.plot(t_plot[:15000],plt_laser[i][:15000],'r')
# plt.xlabel("nano seconds")
# plt.show()

# fig = plt.figure(figsize=(15,15))
# fig.suptitle("Transimpedenance Amplifier - High Cut-Off Frequency", fontsize=26)

# for i in range(1):
#     ax = plt.subplot(4,1,2)
#     red_patch = mpatches.Patch(color='tab:orange', label='Unsychronized Digitizer')
#     green_patch = mpatches.Patch(color='tab:cyan', label='Sychronized Digitizer')
#     plt.legend(handles=[red_patch,green_patch])
#     ax.plot(t_plot[:15000],plt_unsync_sample_loc[i][:15000],'tab:orange')
#     ax.plot(t_plot[:15000],plt_sync_sample_loc[i][:15000],'tab:cyan')
#     ax.plot(t_plot[:15000],plt_highbw_transimpndance[i][:15000],'b')
# plt.xlabel("nano seconds")
# plt.show()
max(avg_cc)

