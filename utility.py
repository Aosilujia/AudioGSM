import numpy as np
import matplotlib.pyplot as plt

#
def slidingwindow(X = np.array([]), n = 1, p = 0):
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

#combine the buffered array together, only need overlap p as param, length could be obtained from X
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

#find cross-correlation peaks
def findpeaks(data,sample_length):
    corrdata=np.abs(data)
    npsignalplot(corrdata)
    avg=np.mean(corrdata)
    std=np.std(corrdata)
    """calculate expected frame number, by changing threshold improve to the expected number"""
    ideal_frame_number=2*int(data.size/sample_length)
    detection_number=data.size #maximum
    threshold=avg+std
    while detection_number>=ideal_frame_number:
        peaks_position=np.where(corrdata>threshold)
        detection_number=np.size(np.where(np.diff(peaks_position)>20))
        threshold+=std
    print("picking correlation result upper than",threshold)
    print((peaks_position))
    print(np.size(peaks_position))
    return peaks_position

def findpeak(data):
    return np.argmax(data)

#return cross-correlation peak number
def peakcount(corrdata,sample_length):
    return np.size(findpeaks(corrdata,sample_length))

#pick a peak as frame start
def pickframe(corrdata,sample_length):
    #peaks_position=findpeaks(corrdata,sample_length)
    peak_position=findpeak(corrdata)
    #choose the first peak by minus sample_length
    first_peak=peak_position-(int(peak_position/sample_length-2))*sample_length
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

if __name__ == '__main__':
    A=[1,2,3,4,5,6,7,8,9,10,11]
    #B=slidingwindow(A,9,1)
    #print(B)
    #C=reshape_add(B,1)
    #print(C)
