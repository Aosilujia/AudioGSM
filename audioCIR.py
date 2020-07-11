import sys
import argparse
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
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from utility import *
from pcm2wav import pcm2wav


"""program control flags """

PRECISE_FRAME_DETECTION_MODE=0 #don't set precise mode to 1, it is shit

""" The sampling rate of the analog to digital convert """
sampling_rate_send=44100.0  #sampling rate of the sending signal
sampling_rate = 48000.0     #sampling rate of the received signal
amplitude=4                 #amplitude parameter, changes the amplitude
amplitude_wav = 24000       #sending signal amplitude parameter, only for wav lib

genfile = "test.wav"

nparrayfile="sample4receive.npy"

"""audio params """
frequecy_center=20000 #18kHz-22kHz
num_samples = 48000
channel_bandwidth=4000
frequency_lowerbound=frequecy_center-channel_bandwidth/2
frequency_higherbound=frequecy_center+channel_bandwidth/2

nframes=num_samples
comptype="NONE"
compname="not compressed"
nchannels=1
sampwidth=2

""" sequnce propertys"""
sequnce_length=26
interval_length=38  #number of interval zeros between sequences
P=16    #cir P, P must be bigger than L
L=sequnce_length-P
circulate_time=7500 #number of frames in the generated audio

"""sequences """
GTS=[0,0,1,0,0,1,0,1,1,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1,1,1]

GTS_2=[0,0,1,0,1,1,0,1,1,1,0,1,1,1,1,0,0,0,1,0,1,1,0,1,1,1]

GTS_3=[0,1,0,0,0,0,1,1,1,0,1,1,1,0,1,0,0,1,0,0,0,0,1,1,1,0]

GTS_4=[0,1,0,0,0,1,1,1,1,0,1,1,0,1,0,0,0,1,0,0,0,1,1,1,1,0]

GTS_5=[0,0,0,1,1,0,1,0,1,1,1,0,0,1,0,0,0,0,0,1,1,0,1,0,1,1]

GTS_6=[0,1,0,0,1,1,1,0,1,0,1,1,0,0,0,0,0,1,0,0,1,1,1,0,1,0]

GTS_7=[1,0,1,0,0,1,1,1,1,1,0,1,1,0,0,0,1,0,1,0,0,1,1,1,1,1]

GTS_8=[1,1,1,0,1,1,1,1,0,0,0,1,0,0,1,0,1,1,1,0,1,1,1,1,0,0]

Barker=[-1,-1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1]

Barker_11=[1,1,1,0,0,0,1,0,0,1,0]

"""program parames"""
DEBUG_MODE=0


def hammingwindow(data):
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

    """zero-padding
        产生两份样本，一份是发送的采样率，一份是接收的采样率
    """
    signal_unpadded=fft_sequence

    upsample_length=sampling_rate_send/channel_bandwidth
    upsample_length_receive=sampling_rate/channel_bandwidth

    #add zero in middle of signal插零要在fft得到信号中间
    insert_position=int(signal_unpadded.size/2)
    zero_number=int(upsample_length*signal_unpadded.size-signal_unpadded.size)
    signal_zero_padded=np.insert(signal_unpadded,insert_position,np.zeros(zero_number))

    zero_number_receive=int((upsample_length_receive-1)*signal_unpadded.size)
    signal_zero_padded_receive=np.insert(signal_unpadded,insert_position,np.zeros(zero_number_receive))

    """ifft"""
    signal_iffted=fftp.ifft(signal_zero_padded)
    signal_iffted_receive=fftp.ifft(signal_zero_padded_receive)





    """save correlation sample"""
    #np.save(nparrayfile,signal_iffted_receive)

    np.save(nparrayfile,signal_iffted_receive)
    selfCorr(signal_iffted_receive)


    """change frequency    """
    cosine_wave = np.array([np.cos(2 * np.pi * frequecy_center/sampling_rate_send * x) for x in range(signal_iffted.size)])*math.sqrt(2)
    signal_inaudible=cosine_wave*signal_iffted




    cosine_wave_receive = np.array([np.cos(2 * np.pi * frequecy_center/sampling_rate * x) for x in range(signal_iffted_receive.size)])*math.sqrt(2)
    signal_inaudible_receive=cosine_wave_receive*signal_iffted_receive

    signal_inaudible*=amplitude

    signal_inaudible_receive*=amplitude


    #   np.save(nparrayfile,signal_inaudible_receive)

    """highpass filtering"""
    b, a = signal.butter(8, 2*(frequency_lowerbound-100)/sampling_rate_send, 'highpass')
    filteredData = signal.filtfilt(b, a, signal_inaudible)


    """add window"""
    windowed_data=hammingwindow(filteredData)
    #npsignalplot(signal_iffted)

    #np.save(nparrayfile,windowed_data)
    return windowed_data


def generate():
    final_data=[]
    """GSM training sequence"""

    signal_inaudible=upsample(GTS)
    """circulate GSM sequnce, change one byte at a time"""
    for i in range(circulate_time):
        final_data=np.append(final_data,signal_inaudible)

    #npsignalplot(final_data)

    """set wave file params and write data
        wav lib does not support float number, so use scipy.io.wavfile is better
    """

    """#DEPRECATED
    wav_file=wave.open(genfile, 'w')
    wav_file.setparams((nchannels, sampwidth, int(sampling_rate_send), int(sampling_rate_send), comptype, compname))
    #print("highest Amp =",max(final_data))
    for s in final_data:
       wav_file.writeframes(struct.pack('h', int(s*amplitude_wav)))
    """

    final_data=np.float32(final_data) #wavfile support float64,but most smartphone only support float32
    wavfile.write(genfile,int(sampling_rate_send),final_data)

    return 1

"""denoise with high pass filtering"""
def denoise_HPF(data):
    """"""
    filtrate=2*(frequency_lowerbound-1000)/sampling_rate
    b,a=signal.butter(8,filtrate,'highpass')
    filteredData=signal.filtfilt(b,a,data)
    return filteredData

"""passband filter"""
def filter2pass(data):
    b, a = signal.butter(8, 2*(2000)/sampling_rate, 'lowpass')
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
    threshold=0.0001;
    #小于一定阈值的都视为无效数据，连续数位超过阈值就进入正常数据部分了
    while i<data.size and continuouscount<1:
        if abs(data[i])<threshold and abs(data[i])>-threshold:
            continuouscount=0
        else:
            continuouscount+=1
        i+=1

    #打印0的数量
    print("zero length:",i)
    return data[i:]

"""the transmitted sequence, maybe used in many cases"""
sequence_Tr=np.zeros(1)
sequence_Tr_inited=0

"""an encapsulated function to get the transmitted sequence"""
def sequenceTr():
    global sequence_Tr
    global sequence_Tr_inited
    if sequence_Tr_inited==0:
        Sequence=(np.load(nparrayfile))
        sequence_Tr=Sequence[:L+P]
        sequence_Tr_inited=1
    return sequence_Tr

"""frame detection, returns the first frame position"""
def framedetection(data,framelength):
    original_array=np.real(np.load(nparrayfile))
    real_data=np.real(data)
    #corr = np.correlate(data,originalarray[:-100], 'valid')
    detect_length=int(framelength*5)
    if (data.size<detect_length):
        detect_length=data.size
    corr=pearsonCroCor(real_data[0:detect_length],original_array)
    #npsignalplot(corr)

    return pickframe(corr,framelength)


""" frame detection precise, returns position array of every frame"""
def framedetection_precise(data,framelength):
    original_array=np.real(np.load(nparrayfile))
    real_data=np.real(data)
    #corr = np.correlate(data,originalarray[:-100], 'valid')
    corr=pearsonCroCor(real_data,original_array)
    #npsignalplot(corr)
    return findpeaks(corr,framelength)


"""Least square way to estimate CIR"""
""" original GTS related"""
array_CTS=np.empty(shape=(1,1))
M_CTS_final=np.empty(shape=(1,1))
matrix_CTS_inited=0



"""
    CIR matrix calculation, the return matrix M size is L*P, and CIR=M*y
    refer to 'Channel Estimation Modeling' from Nokia Research Center for details
"""
def CIR_Matrix(L_L,P_L,Seq):
    global matrix_CTS_inited
    global array_CTS

    if matrix_CTS_inited==0 or array_CTS.shape[0]!=P_L:
        """create circulant training sequence matrix"""
        array_CTS=np.empty(shape=(P_L,L_L+1))
        for i in range(P_L):
            line=Seq[i:i+L_L+1]
            line.reverse()
            array_CTS[i]=np.asarray(line)
        M_CTS=np.matrix(array_CTS)
        """calculate M"""
        print(M_CTS)
        print(((M_CTS.H) * M_CTS ))
        M_CTS_final=((M_CTS.H*M_CTS).I)*M_CTS.H
        print(M_CTS_final)
        print(1/P_L * M_CTS.H)
        matrix_CTS_inited=1
    """do matrix translation"""
    #result_CIR=M_CTS_final*np.matrix(data[L_L:L_L+P_L]).T
    #print(result_CIR)
    return M_CTS_final



""" expanding sequence related"""
circulant_TSM_expanding_inited=0
circulant_TSM_expanding=np.empty(shape=(1,1))
M_CTS_expanding_final=np.empty(shape=(1,1))
matrix_CTS_expanding_inited=0;



"""
    CIR estimation from expanded GSM
"""

"""the matrix 'M' """
def Circulant_TSM(L_L,P_L,expanding_length):
    global circulant_TSM_expanding_inited
    global circulant_TSM_expanding
    L=L_L*expanding_length
    P=P_L*expanding_length
    if circulant_TSM_expanding_inited==0 or (circulant_TSM_expanding.shape[0]!=P or circulant_TSM_expanding.shape[1]!=L+1):
        circulant_TSM_expanding=np.empty(shape=(P,L+1),dtype=complex)
        #存的如果是ifft回来的结果为复数
        Sequence=(np.load(nparrayfile))
        for i in range(P):
            line=Sequence[i:i+L+1]
            line=np.flip(line,0)
            circulant_TSM_expanding[i]=line
    return circulant_TSM_expanding



"""(M^H M)^-1 M^H"""
def CIR_Matrix_Expanding(L_L,P_L,expanding_length):
    global matrix_CTS_expanding_inited
    global M_CTS_expanding_final

    L=L_L*expanding_length
    P=P_L*expanding_length
    if matrix_CTS_expanding_inited==0 or (M_CTS_expanding_final.shape[0]!=L+1 or M_CTS_expanding_final.shape[1]!=P):
        M_CTS=np.matrix(Circulant_TSM(L_L,P_L,expanding_length))
        M_CTS_expanding_final=((M_CTS.H*M_CTS).I)*M_CTS.H
        #print((M_CTS.H*M_CTS)[0])
        matrix_CTS_expanding_inited=1
    return M_CTS_expanding_final

def CIR_Matrix_Expanding_approximation(L_L,P_L,expanding_length):
    global matrix_CTS_expanding_inited
    global M_CTS_expanding_final

    M_CTS=np.matrix(Circulant_TSM(L_L,P_L,expanding_length))
    L=L_L*expanding_length
    P=P_L*expanding_length
    if matrix_CTS_expanding_inited==0 or (M_CTS_expanding_final.shape[0]!=L+1 or M_CTS_expanding_final.shape[1]!=P):
        M_CTS_expanding_final=M_CTS.H
        matrix_CTS_expanding_inited=1
    return M_CTS_expanding_final

"""cross-correlation sample for cir estimation"""
def CIR_Sample_CC(L_L,P_L,expanding_length):
    return sequenceTr()


"""use original 26bit GTS to generate CIR"""
def CIR_LS_SIMPLE(data):
    M_CTS=CIR_Matrix(L,P,GTS)
    result_CIR=M_CTS*np.matrix(data[L:L+P]).T
    return result_CIR

"""use expanded GTS to generate longer CIR"""
def CIR_LS_Expanding(data,expanding_length):
    M_translation=CIR_Matrix_Expanding_approximation(L,P,expanding_length)
    result_CIR=M_translation*((np.matrix(data[L*expanding_length:(L+P)*expanding_length])).T)
    return result_CIR.flatten()[0].tolist()

"""cross-correlation way of CIR estimation """
def CIR_CC_Expanding(data,expanding_length):
    sample_send=CIR_Sample_CC(L,P,expanding_length)
    result_CIR=np.correlate(data[:(L+P)*expanding_length],sample_send,'same')
    return result_CIR.tolist()


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

    #samplerate, wave_data = wavfile.read(filename) #


    #npsignalplot(wave_data)


    """pre-processing remove zero in the front"""
    filtered_Data=denoise_HPF(removefrontzero(wave_data))
    #npsignalplot(filteredData)


    """call downsample function"""
    downsampled_signal=downsample(filtered_Data)
    #npsignalplot(downsampled_signal)
    """frame detection"""

    expanding_length=int(sampling_rate/channel_bandwidth)   #the expanding length for every bit
    effective_length=int(sequnce_length*expanding_length)  #the gsm part length

    frame_length=(sequnce_length+interval_length)*expanding_length  #the whole frame length, including gsm and zeros
    practical_frame_length=int((sequnce_length+interval_length)*(sampling_rate_send/channel_bandwidth)) /sampling_rate_send * sampling_rate  #the real frame length may not be precise because of 44.1k to 48k

    print("frame length=",frame_length)
    print("practical frame length=",practical_frame_length)


    downsampled_signal=downsampled_signal[frame_length*10:]#remove the front frames in case of hardware jaming


    framepos=framedetection(downsampled_signal,practical_frame_length) #the first frame position
    print("first frame position=",framepos)

    """-estimate cir based on frame detection """


    frame_datas=downsampled_signal
    #ready to write to csv file
    csvfile=filename[:filename.rfind(".")]+".csv"
    f=open(csvfile,'w',encoding='utf-8',newline='')
    csv_writer=csv.writer(f)

    """     iterate to calculate each frame cir"""

    framecount=0


    if PRECISE_FRAME_DETECTION_MODE==1:
        peaks=framedetection_precise(frame_datas,practical_frame_length)
        print ("peak count:",peaks.size)
        print(peaks)
        for peak in peaks:
            framesequence=frame_datas[int(peak):int(peak)+effective_length]
            """CIR estimation for each frame"""
            npsignalplot(framesequence)
            cir=CIR_LS_Expanding(framesequence,expanding_length)
            csv_writer.writerows(cir)

    elif PRECISE_FRAME_DETECTION_MODE==0:
        i=int(framepos)
        while i+frame_length<=np.size(frame_datas):
            framesequence=frame_datas[int(i):int(i)+effective_length]
            """CIR estimation for each frame"""
            cir=CIR_LS_Expanding(framesequence,expanding_length)

            csv_writer.writerows(cir)
            """relocate frame process"""
            """
            framecount+=1
            if framecount>=10:
                framepos=framedetection(frame_datas[int(i):],practical_frame_length)
                if (framepos>=frame_length/2):
                    i-=practical_frame_length
                i+=framepos
                print("aligned position:",framepos)
                framecount=0
            """
            i+=practical_frame_length  #choose practical frame length to avoid mistakes


"""estimate files from a filepath or file, altered from pcm2wav"""
def estimatefiles(path='.'):
    """先pcm转wav"""
    pcm2wav(path)
    """"""
    if path.endswith('.pcm'):
        path=path[:-4]+'.wav'

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


"""generate sending signal with modified parameters"""
def generate2():
    print(1+1)

def estimate2(s='./'):
    print(s)

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
                "-c,cir [filename]:distract csi from received audio file or files\n"
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
        sequence_data=GTS
        for i in range(len(sequence_data)):
            if (sequence_data[i]==0):
                sequence_data[i]=-1
        print(sequence_data)
        matrix=CIR_Matrix(5,8,sequence_data)
        #matrix=(1/(P*12))*np.matrix(Circulant_TSM(L,P,12)).H
        matrix=CIR_Matrix_Expanding_approximation(L,P,12)
        matrix2=np.zeros(matrix.shape[0])
        for i in range(matrix.shape[0]):
            matrix2[i]=np.sum(matrix[i])
        print(sum(matrix2))
        npsignalplot(matrix2)

        #generate()
        #estimate("1_honor.wav")
        #estimatefiles("2020-05-05-14-03-02")
