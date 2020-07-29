import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr


"""pearson cross correlation"""
def pearsonCroCor(data,sample):
    corresult=np.zeros(data.size-sample.size+1)
    #slide the data, every time pick sample.size elements to do pearson with sample
    i=0
    sample_size=sample.size
    while i+sample_size<=data.size:
        #corresult[i]=pearsonr(data[i:i+sample_size],sample)[0]
        corresult[i]=np.corrcoef(data[i:i+sample_size],sample)[0][1]
        i+=1
    return corresult

"""test the self-correlation properties of one data """
def selfCorr(data):
    npsignalplot(pearsonCroCor(np.append(np.append(data,data),data),data))

""" sliding window"""
def slidingwindow(X = np.array([]), n = 1, p = 1):
    #buffers data vector X into length n column vectors with overlap p
    #excess data at the end of X is discarded
    n = int(n) #length of each data vector
    p = int(p) #overlap of data vectors, 0 <= p < n-1
    L = np.size(X) #length of data to be buffered
    m = int(np.floor((L-n)/p) + 1) #number of sample vectors (no padding)
    data = np.zeros([m,n]) #initialize data matrix
    for startIndex,row in zip(range(0,L-n+1,p),range(0,m)):
        data[row] = X[startIndex:startIndex + n] #fill in by column
    return data

"""combine the buffered array together, only need overlap p as param, length could be obtained from X"""
def reshape_add(X = np.array([]),p=0):
    n = X.shape[1]
    p = int(p)
    datasize=n+p*(len(X)-1)
    subarray_number=len(X)
    data = np.zeros(datasize)
    for i in range(subarray_number):
        if i*p+n>datasize:
            data[i*p:]+=X[i][:(datasize-i*p)]
            break
        data[i*p:i*p+n]+=X[i]
    return data

"""find all correlation peaks"""
def findpeaks(corrdata,sample_length):
    peak_positions=[]
    npsignalplot(corrdata)
    i=0
    """note: divide by 2 is corresponding to the length of gsm and zeros added
    """
    while i+sample_length/2<=corrdata.size:
        if int(i+sample_length)>=corrdata.size:
            i=int(i)+np.argmax(np.abs(corrdata[int(i):]))
        else:
            i=int(i)+np.argmax(np.abs(corrdata[int(i):int(i+sample_length)]))
        peak_positions.append(int(i))
        i+=sample_length/2
    return np.asarray(peak_positions)

"""find the peak"""
def findpeak(data):
    return np.argmax(np.abs(data))

"""return cross-correlation peak number"""
def peakcount(corrdata,frame_length):
    return np.size(findpeaks(corrdata,frame_length))

"""calculate an early peak based on the result"""
def pickframe(corrdata,frame_length):
    peak_position=findpeak(corrdata)
    """choose the first peak by minus length of frames
        the first int() is for the integer part of dividing, the second int() is used in case that frame_length is not integer
    """
    first_peak=peak_position-int(int(peak_position/frame_length)*frame_length)
    return first_peak


#fast signal pyplot with np array
def npsignalplot(data):
    plt.figure()
    plt.plot(np.arange(data.size),data)
    plt.show()

#fast signal pyplot with python list
def listsignalplot(data):
    plt.figure()
    plt.plot(np.arange(len(data)),data)
    plt.show()

#fft result plot
def fftplot(fft_y):
    x = np.arange(fft_y.size)
    abs_y=np.abs(fft_y)                # 取复数的绝对值，即复数的模(双边频谱)
    angle_y=np.angle(fft_y)              #取复数的角度

    plt.figure()
    plt.plot(x,abs_y)
    plt.title('双边振幅谱（未归一化）')

    plt.figure()
    plt.plot(x,angle_y)
    plt.title('双边相位谱（未归一化）')
    plt.show()

#根据需要的音频时长（秒）和单帧时长（长度和采样率计算）计算帧的数量
def framenumber(time,framelength,samplerate):
    frametime=framelength/samplerate
    return int(time/frametime)



if __name__ == '__main__':
    A=[1,2,3,4,5,6,7,8,9,10,11]
    #B=slidingwindow(A,9,1)
    #print(B)
    #C=reshape_add(B,1)
    #print(C)
