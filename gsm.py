import sys
import numpy as np
import math
import wave
import struct
import csv
import os
import queue
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import scipy.fftpack as fftp
import matplotlib.pyplot as plt
from utility import pickframe,slidingwindow,reshape_add,findpeaks,npsignalplot,listsignalplot,fftplot

""" params """
frequecy_center=20000 #18kHz-22kHz
num_samples = 48000
channel_bandwidth=4000
frequency_lowerbound=frequecy_center-channel_bandwidth/2
frequency_higherbound=frequecy_center+channel_bandwidth/2

""" The sampling rate of the analog to digital convert """
sampling_rate_send=44100.0
sampling_rate = 48000.0
amplitude = 16000

file = "test.wav"
nparrayfile="singlesignal.npy"
nframes=num_samples
comptype="NONE"
compname="not compressed"
nchannels=1
sampwidth=2

""" sequnce propertys"""
sequnce_length=26
interval_length=38
P=16
L=sequnce_length-P
circulate_time=4000

GTS=[0,0,1,0,0,1,0,1,1,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1,1,1]

sine_wave=[np.sin(2 * np.pi * x/20) for x in range(100)]


def createwave(filename,data):
    """set wave file params and write data"""
    wav_file=wave.open(filename, 'w')
    wav_file.setparams((nchannels, sampwidth, int(sampling_rate), nframes, comptype, compname))
    for s in data:
       wav_file.writeframes(struct.pack('h', int(s*amplitude)))

def hammingwindow(data):
    #length_window=100
    #length_overlap=length_window/2
    length_window=np.size(data)
    length_overlap=length_window
    #split data with a sliding window
    splitdata=slidingwindow(data,length_window,length_overlap)
    #each row multiply with hamming
    hamming_window=np.hamming(length_window)
    for i in range(len(splitdata)):
        splitdata[i]*=hamming_window
    return reshape_add(splitdata,length_overlap)

#upsample process
def upsample(sequence):
    """set the 0 in sequence to -1"""
    sequence_data=sequence
    for i in range(len(sequence_data)):
        if (sequence_data[i]==0):
            sequence_data[i]=-1

    sequence_signal=np.asarray(sequence_data)

    """add interval zeros"""
    sequence_with_space=np.append(sequence_signal,np.zeros(interval_length))

    """fft fast  fourier transform"""
    fft_sequence=fftp.fft(sequence_with_space)

    #fftplot(fft_sequence)

    """zero-padding"""
    upsample_length=int(sampling_rate_send/channel_bandwidth)
    upsample_length_receive=int(sampling_rate/channel_bandwidth)

    signal_unpadded=fft_sequence

    #add zero in middle of signal
    insert_position=int(signal_unpadded.size/2)
    zero_number=upsample_length*signal_unpadded.size-signal_unpadded.size
    signal_zero_padded=np.insert(signal_unpadded,insert_position,np.zeros(zero_number))

    zero_number_receive=int((upsample_length_receive-1)*signal_unpadded.size)
    signal_zero_padded_receive=np.insert(signal_unpadded,insert_position,np.zeros(zero_number_receive))

    #fftplot(signal_zero_padded)

    """ifft"""
    signal_iffted=fftp.ifft(signal_zero_padded)
    signal_iffted_receive=fftp.ifft(signal_zero_padded_receive)

    """add window"""
    windowed_data=hammingwindow(signal_iffted)
    npsignalplot(windowed_data)
    #npsignalplot(signal_iffted)

    """save correlation sample"""
    np.save(nparrayfile,hammingwindow(signal_iffted_receive))

    """change frequency    """
    cosine_wave = np.array([np.cos(2 * np.pi * frequecy_center/sampling_rate_send * x) for x in range(signal_iffted.size)])*math.sqrt(2)
    signal_inaudible=cosine_wave*windowed_data

    #npsignalplot(signal_inaudible)

    """bandpass filtering"""
    b, a = signal.butter(8, 2*(frequency_lowerbound-1000)/sampling_rate_send, 'highpass')
    filteredData = signal.filtfilt(b, a, signal_inaudible)

    return filteredData


def generate():
    final_data=[]
    """GSM training sequence"""

    signal_inaudible=upsample(GTS)
    """circulate GSM sequnce, change one byte at a time"""
    for i in range(circulate_time):
        #listsignalplot(GTS)
        final_data=np.append(final_data,signal_inaudible)

    #npsignalplot(final_data)
    """set wave file params and write data"""
    wav_file=wave.open(file, 'w')
    wav_file.setparams((nchannels, sampwidth, int(sampling_rate_send), int(sampling_rate_send), comptype, compname))
    #print("highest Amp =",max(final_data))
    for s in final_data:
       wav_file.writeframes(struct.pack('h', int(s*amplitude)))

    return final_data

"""denoise with high pass filtering"""
def denoise_HPF(data):
    """"""
    filtrate=2*(frequency_lowerbound-1000)/sampling_rate
    print(filtrate)
    b,a=signal.butter(8,filtrate,'highpass')
    filteredData=signal.filtfilt(b,a,data)
    return filteredData

"""passband filter"""
def filter2pass(data):
    b, a = signal.butter(8, 2*(frequency_lowerbound-1000)/sampling_rate, 'highpass')
    filteredData = signal.filtfilt(b, a, data)
    return filteredData


"""downsample"""
def downsample(data):
    realpart=data*np.array([np.cos(2 * np.pi * frequecy_center/sampling_rate * x) for x in range(data.size)])*math.sqrt(2)
    imaginarypart=data*np.array([np.sin(2 * np.pi * frequecy_center/sampling_rate * x) for x in range(data.size)])*math.sqrt(2)
    #return realpart+imaginarypart*-1j
    return filter2pass(realpart)+filter2pass(imaginarypart)*-1j

"""remove zero in front from np array"""
def removefrontzero(data):
    i=0
    continuouscount=0;
    threshold=0.001;
    while i<data.size and continuouscount<3:
        if abs(data[i])<threshold and abs(data[i])>-threshold:
            continuouscount=0
        else:
            continuouscount+=1
        i+=1
    return data[i:]

"""pearson cross correlation"""
def pearsonCroCor(data,sample):
    corresult=np.zeros(data.size-sample.size+1)
    #slide the data, every time pick sample.size elements to do pearson with sample
    i=0
    while i+sample.size<=data.size:
        corresult[i]=np.corrcoef(data[i:i+sample.size],sample)[0,1]
        i+=1
    return corresult

"""frame detection"""
def framedetection(data):
    originalarray=np.load(nparrayfile)
    npsignalplot(originalarray)
    #corr = np.correlate(data,originalarray[:-100], 'valid')
    corr=pearsonCroCor(data[0:20000],originalarray)
    #print(originalarray.size)
    #print(corr.size)
    npsignalplot(corr)
    return pickframe(corr,originalarray.size)
    #return 0

"""Least square way to estimate CIR"""
array_CTS=np.empty(shape=(P,L))
M_CTS_final=np.empty(shape=(L,P))
matrix_CTS_inited=0
"""
    CIR matrix calculation, the return matrix M size is L*P, and CIR=M*y
    refer to 'Channel Estimation Modeling' from Nokia Research Center for details
"""
def CIR_Matrix(L_L,P_L,Seq):
    global matrix_CTS_inited

    if matrix_CTS_inited==0:
        """create circulant training sequence matrix"""
        for i in range(P):
            line=Seq[i:i+L_L]
            line.reverse()
            array_CTS[i]=np.asarray(line)
        M_CTS=np.matrix(array_CTS)
        """calculate M"""
        M_CTS_final=(M_CTS.H*M_CTS).I*M_CTS.H
        array_CTS_inited=1
    """do matrix translation"""
    #result_CIR=M_CTS_final*np.matrix(data[L_L:L_L+P_L]).T
    #print(result_CIR)
    return M_CTS_final

"""use original 26bit GTS to generate CIR"""
def CIR_SIMPLE(data):
    M_CTS=CIR_Matrix(L,P,GTS)
    result_CIR=M_CTS*np.matrix(data[L:L+P]).T
    return result_CIR

"""use expanded GTS to generate longer CIR"""
def CIR_Expanding(data,expanding_length):
    M_original=CIR_Matrix(L,P,GTS)
    data_matrix=np.matrix(data[L*expanding_length:(L+P)*expanding_length]).reshape((P,expanding_length))
    return data_matrix

"""extract sequence from frame"""
def received_sequence(data):
    resultsequence=[]
    upsample_length=int(sampling_rate/channel_bandwidth)
    i=0
    while i+upsample_length<=len(data):
        bit_data=data[i:i+upsample_length]
        avg_bit=np.sum(np.asarray(bit_data))/upsample_length
        resultsequence.append(avg_bit)
        i+=upsample_length
    return resultsequence


"""main function to estimate received signal"""
def estimate(filename):

    """read the wave file raw data
    wave lib cannot support 32bit float, but 32bit float samplewidth could be read by np.frombuffer() parameter dtype
    """
    wav_file=wave.open(filename,'rb')
    params = wav_file.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    str_data = wav_file.readframes(nframes)
    wav_file.close()

    """convert raw data
    dtype correspond to samplewidth, 2byte=short, 32bit float=single
    """
    wave_data = np.frombuffer(str_data, dtype=np.single)
    npsignalplot(wave_data)
    #fftplot(fftp.fft(wave_data))

    """denoise"""
    #filteredData=denoise_HPF(wave_data)
    #npsignalplot(filteredData)

    """remove zero in the front"""
    filteredData=removefrontzero(wave_data)
    #npsignalplot(filteredData)

    """call downsample function"""
    downsampled_signal=downsample(filteredData)
    #npsignalplot(downsampled_signal)
    #createwave('test2.wav',filteredData)
    """frame detection"""
    framepos=framedetection(downsampled_signal) #the first frame position
    print("first frame position=",framepos)

    """-estimate cir based on frame detection """
    expanding_length=int(sampling_rate/channel_bandwidth)
    frame_length=(sequnce_length+interval_length)*expanding_length  #the whole frame length, including gsm and zeros
    effective_length=int(sequnce_length*expanding_length)  #the gsm part length
    print("frame length=",frame_length)
    frame_datas=downsampled_signal
    """     ready to write to csv file"""
    csvfile=filename[:filename.rfind(".")]+".csv"
    f=open(csvfile,'w',encoding='utf-8',newline='')
    csv_writer=csv.writer(f)
    """     iterate to calculate each frame cir"""
    i=int(framepos)
    while i+frame_length<=np.size(frame_datas):
        framesequence=frame_datas[i:i+effective_length]
        """CIR estimation for each frame"""
        cir=CIR_Expanding(framesequence,expanding_length)
        csv_writer.writerows(cir.flatten()[0].tolist())
        i+=frame_length

"""estimate files from a filepath or file, altered from pcm2wav"""
def estimatefiles(path='.'):
    if path.endswith('.wav'):
        estimate(path)
        return
    filequeue = queue.Queue()
    filequeue.put(path)
    while not filequeue.empty():
        node = filequeue.get()
        for filename in os.listdir(node):
            nextpath = os.path.join(node, filename)
            if os.path.isdir(nextpath):
                filequeue.put(nextpath)
            elif nextpath.endswith('.wav'):
                """"""
                estimate(nextpath)


if __name__ == '__main__':
    #script params
    params=sys.argv
    if (len(params)>1):
        i=1
        """"params"""
        while i<len(params):
            arg=params[i]
            i+=1
            if arg=='-h' or arg=='h':
                print("Usage:\n",
                "-h:list the usage\n",
                "-g,generate:generate gsm-based audio as wav file,default name is ",file,"\n"
                "-c,cir:distract csi from received audio file or files\n"
                )
            elif arg=='-g' or arg=='generate':
                print("generating")
                generate()
            elif arg=='-c' or arg=='cir':
                if (i<len(params)):
                    path=params[i]
                    print("distracting cir of ",path)
                    estimatefiles(path)
                else:
                    print("-c or cir need a param of file or directory name")
    else:
        """manual debug"""
        #generate()
        estimate("6_441.wav")
        #estimatefiles("2020-05-05-14-03-02")
