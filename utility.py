import numpy as np
import os
import csv
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
    return pearsonCroCor(np.append(data,data),data)

def figureSelfCorr(data,label='Sequence name'):
    fig, ax = plt.subplots()  # Create a figure and an axes.
    corrresult=selfCorr(data)
    x = np.linspace(0, corrresult.size, corrresult.size)
    ax.plot(x, corrresult, label=label,color='orange')  # Plot some data on the axes.
    ax.set_xlabel('Sequence Index')  # Add an x-label to the axes.
    ax.set_ylabel('PCC')  # Add a y-label to the axes.
    ax.legend()  # Add a legend.
    plt.show()

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
    #npsignalplot(corrdata)
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


"""write multiple cir data to disk, input should be a list of cir data"""
def writeCIRs2csv(cirs,tag="",output_path="./tmpCIR"):
    """check directory"""
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    """check any filename that matches the tag, to generate an id"""
    data_count=0
    for root, dirs, files in os.walk(output_path):
        """walk for all the filenames"""
        for filename in files:
            """check if the filename contains the tag"""
            if (len(filename)>len(tag)):
                if (filename[:len(tag)]==tag):
                    """"check the next character after the tag to make sure this is not another tag with the same former part"""
                    if (filename[len(tag):len(tag)+1]=="_"):
                        data_count+=1

    """base file path to add numbers"""
    base_path=output_path+"/"+tag+"_"

    """write every cir to disk, increment id every time"""
    for cir in cirs:
        file_path=base_path+str(data_count)+'.csv'
        data_count+=1
        CIR2csv(cir,file_path)


"""write cir data to disk, would check the tag name for a unique filename"""
def writeCIR2csv(cir,tag="",output_path="./tmpCIR",distance=""):
    """check directory"""
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    """check any filename that matches the tag, to generate an id"""
    data_count=0
    for root, dirs, files in os.walk(output_path):
        """walk for all the filenames"""
        for filename in files:
            """check if the filename contains the tag"""
            if (len(filename)>len(tag)):
                if (filename[:len(tag)]==tag):
                    """"check the next character after the tag to make sure this is not another tag with the same former part"""
                    if (filename[len(tag):len(tag)+1]=="_"):
                        data_count+=1

    """generate the full file path"""
    base_path=output_path+"/"+tag+"_"
    data_name=base_path+str(data_count)
    if (distance!=""):
        data_name+='_'+distance
    file_path=data_name+'.csv'

    """write to disk"""
    CIR2csv(cir,file_path)

"""SHALL NOT BE IMPORTED write one cir data to csv, by precise file name and directory,without checking,"""
def CIR2csv(cir,file_path=""):
    f = open(file_path, 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerows(cir)
    f.close()

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
