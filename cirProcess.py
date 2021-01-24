import csv
import os
import re
import threading
from datetime import datetime
import numpy as np
import scipy.signal
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from utility import npsignalplot,listsignalplot,slidingwindow,writeCIR2csv

filename = "gsmtest/fixed/8.csv"



# filename="test.csv"

#-------------------------------------------
"""
    unfinished part:about distracting distance info from cir
"""
def tap2cm(tap):
    return (1/48000)*343*tap/2*100


def most_effected_taps(dcir,tap_number=10):
    cir_diff=dcir
    taps_locations=[]
    count=0
    """目前是固定阈值，可以添加参数每次计算"""
    static_cir_avg=0.0005533
    for (index,cir) in enumerate(cir_diff):
        kth=-tap_number
        cir_maxargs=np.argpartition(cir,kth)[kth:]
        while np.sum(cir_maxargs>100)>0:
            np.put(cir,cir_maxargs[cir_maxargs>100],0)
            cir_maxargs=np.argpartition(cir,kth)[kth:]
        while np.sum(cir_maxargs<5)>0:
            np.put(cir,cir_maxargs[cir_maxargs<5],0)
            cir_maxargs=np.argpartition(cir,kth)[kth:]
        cir_maxavg=np.mean(cir[cir_maxargs])
        if (cir_maxavg>static_cir_avg*4):
            taps_locations.append(cir_maxargs.tolist())
        else:
            taps_locations.append(np.zeros(tap_number).tolist())
        #print(index,(cir_maxavg/static_cir_avg))
    return taps_locations

def tap_locations_clustering(tap_locations):
    clustering_result=[]
    clustering_numbers=[]
    for tap in tap_locations:
        if len(clustering_result)==0:
            clustering_result.append(tap)
            clustering_numbers.append(1)
            continue
        flag_found=0
        for (index,cluster) in enumerate(clustering_result):
            if (cluster-tap)<=10 and (cluster-tap)>=-10:
                cluster_num=clustering_numbers[index]
                clustering_numbers[index]+=1
                clustering_result[index]=(clustering_result[index]*cluster_num+tap)/(cluster_num+1)
                flag_found=1
                break
        if (flag_found==1):
            continue
        clustering_result.append(tap)
        clustering_numbers.append(1)
    return clustering_result

#-------------------------------------------
"""cir cutting functions"""
"""
通过检测数据变动较大的时间点（波峰），判断波峰处为有效数据（手势或写字），以波峰为中心取多段数据
"""
def partition_word_bypeak(cir_file,tag_content="", output_path="./training_data",writecsv=False,data_length=200,show_charts=False,lengthen_rule={},static_threshold=0.02,desired_peaks=1):
    """读数据"""
    cir_data = np.genfromtxt(cir_file, dtype=complex, delimiter=',')
    """作差"""
    cir_diff = np.abs(np.diff(cir_data, axis=0))
    """对作差的数据求和"""
    dcir_sum = np.sum(cir_diff, axis=1)
    """平滑数据曲线"""
    Mguassian=gaussian_filter1d(dcir_sum * dcir_sum, 6)

    """根据标签长度延长数据段长度"""
    if (len(tag_content)>=3):
        data_length+=50*(len(tag_content)-2)

    """根据data length的参数，默认从peak向左向右各取一半长度"""
    righten_length=int(data_length/2)
    leften_length=int(data_length/2)

    """有排序规则，读排序规则对应的字母，确定每个字母对应的延长规则"""
    if len(lengthen_rule)>0 and lengthen_rule.has_key(tag_content):
        tag_rulesetting=lengthen_rule.get(tag_content)
        tag_rule=""
        """规则第一位+,-分别表示偏向右还是向左，第二位数字表示偏的长度，没有第二位就取一个默认值"""
        if (len(tag_rulesetting)>=1):
            tag_rule=tag_rulesetting[0]
            if (len(tag_rulesetting)>=2):
                rule_length=tag_rulesetting[1]
                if (rule_length>=data_length):
                    rule_length=data_length
                if (tag_rule=="+"):
                    righten_length=righten_length+int(rule_length/2)
                    leften_length=leften_length-int(rule_length/2)
                elif (tag_rule=="-"):
                    righten_length=righten_length-int(rule_length/2)
                    leften_length=leften_length+int(rule_length/2)
            else:
                if (tag_rule=="+"):
                    righten_length=righten_length+int(data_length/4)
                    leften_length=leften_length-int(data_length/4)
                elif (tag_rule=="-"):
                    righten_length=righten_length-int(data_length/4)
                    leften_length=leften_length+int(data_length/4)

    """取相应数量的数据中心点"""
    base_positions=get_desired_peaks(Mguassian,static_threshold,desired_peaks,righten_length,leften_length)

    if show_charts:
        plt.subplot(2,1,1)
        plt.pcolormesh((np.abs(cir_data)).T)

    dis=1
    """根据得到的左右长度和中心点，取每一个pos对应的区间"""
    for (index,position) in enumerate(base_positions):
        """"""
        print([position-leften_length,position+righten_length])
        if show_charts:
            plt.axvline(x=position-leften_length,ls="-",c="red")
            plt.axvline(x=position+righten_length,ls="-",c="red")
        tag_withinfo=tag_content
        """调整标签以适应CTC，主要是加上字与字之间的转进"""
        if (index==0):
            tag_withinfo=tag_content
        else:
            tag_withinfo="="+tag_content
        if writecsv:
            writeCIR2csv(cir_data[position-leften_length:position+righten_length],tag_withinfo,output_path,distance="{}".format(dis))
        dis+=1

    if show_charts:
        plt.subplot(2,1,2)
        plt.plot(Mguassian)
        plt.show()

    return base_positions

"""已有高斯平滑后的数据，基于输入的阈值进行微调，目标是得到符合（或少于）desired_peaks数量的数据段"""
def get_desired_peaks(Mguassian,static_threshold,desired_peaks,righten_length,leften_length,AUTO_TUNING=True):
    threshold=static_threshold
    threshold_upperbound=threshold*4
    threshold_lowerbound=threshold*0.25
    tuning_gap=threshold/20
    base_positions=[]
    positions_tmp=[]
    tuning_flag=0 #状态码，0表示初始，1表示在上界到中间，-1表示下界到中间
    last_count=0
    auto_TUNING=AUTO_TUNING
    while True:
        positions_tmp=base_positions
        base_positions=[]
        nearby_peaks_pos=[]
        last_peak=0
        """找到所有波峰"""
        peaks=scipy.signal.find_peaks(Mguassian,height=threshold)

        """
            把相邻（距离较近）的peak合并到一起
        """
        for (index,peak) in enumerate(peaks[0]):
            if (peak<leften_length):
                """过早数据剔除"""
                continue
            """如果两个波峰差距过小就视为同一个数据段
                # WARNING:差距数值硬编码注意
            """
            if (peak-last_peak)<=100 or last_peak==0:
                nearby_peaks_pos.append(index)
            else:
                """两个波峰差距超过某值就是为两个数据段，需要处理前面记录在数组中的波峰数据，
                    这些较近波峰要合并为一个，根据波峰高度归一化后成权重，权重乘以位置求加权平均"""
                if (len(nearby_peaks_pos)>1):
                    nearby_peaks=peaks[1].get("peak_heights")[nearby_peaks_pos]
                    weights=nearby_peaks/np.sum(nearby_peaks)
                    base_positions.append(int(np.sum(peaks[0][nearby_peaks_pos]*weights)))
                elif (len(nearby_peaks_pos)==1):
                    base_positions.append(peaks[0][nearby_peaks_pos[0]])
                nearby_peaks_pos=[]
                nearby_peaks_pos.append(index)
            last_peak=peak
        """处理遍历到最后剩下的数据"""
        if (len(nearby_peaks_pos)>1):
            nearby_peaks=peaks[1].get("peak_heights")[nearby_peaks_pos]
            weights=nearby_peaks/np.sum(nearby_peaks)
            if int(np.sum(peaks[0][nearby_peaks_pos]*weights))+righten_length<=Mguassian.size:
                base_positions.append(int(np.sum(peaks[0][nearby_peaks_pos]*weights)))
        elif (len(nearby_peaks_pos)==1):
            if peaks[0][nearby_peaks_pos[0]]+righten_length<=Mguassian.size:
                base_positions.append(peaks[0][nearby_peaks_pos[0]])

        if not auto_TUNING:
            break;

        """记录上一次循环检测到的数量：初始化"""
        if last_count==0:
            last_count=len(base_positions)

        """判断分割的数量和desired_peaks差距"""
        if (len(base_positions)==desired_peaks):
            """数量相符"""
            break
        if (tuning_flag==0):
            """第一次调整时flag为0，判断检测数量和希望数量之间的差距，选择往上或者往下调阈值"""
            if (len(base_positions)<desired_peaks):
                tuning_flag=-1
                threshold-=tuning_gap
            else:
                tuning_flag=1
                threshold+=tuning_gap
        elif (tuning_flag==-1):
            if (len(base_positions)<last_count):
                """往下调阈值检测数量反而更少，不符合常理，判断为自动分割到达极限，回退到上一次结果"""
                threshold+=tuning_gap
                auto_TUNING=False
                continue
            if (len(base_positions)<desired_peaks and threshold<=threshold_lowerbound):
                """阈值调到最小也检不出满足的数量"""
                break
            elif (len(base_positions)>desired_peaks):
                base_positions=positions_tmp
                break
            else:
                threshold-=tuning_gap
        elif(tuning_flag==1):
            if (len(base_positions)<last_count):
                """往上调阈值检测数量反而更多，不符合常理，判断为自动分割到达极限，回退到上一次结果"""
                threshold-=tuning_gap
                auto_TUNING=False
                continue
            if (len(base_positions)>desired_peaks and threshold>=threshold_upperbound):
                """阈值调到最大也检不出满足的数量"""
                break
            elif (len(base_positions)<desired_peaks):
                base_positions=positions_tmp
                break
            else:
                threshold+=tuning_gap
        else:
            break
        """记录上一次循环检测到的数量"""
        last_count=len(base_positions)
    return base_positions

"""手动选取中心坐标，切割cir"""
def partition_cir_manually(cir_file,tag_content="", output_path="./training_data",writecsv=False,data_length=200,lengthen_rule={}):
    """读数据"""
    cir_data = np.genfromtxt(cir_file, dtype=complex, delimiter=',')[:1600]
    """作差"""
    cir_diff = np.abs(np.diff(cir_data.real, axis=0))
    """对作差的数据求和"""
    dcir_sum = np.sum(cir_diff, axis=1)
    """平滑数据曲线"""
    Mguassian=gaussian_filter1d(dcir_sum **2, 8)

    dvalues=[]
    split_data=slidingwindow(Mguassian,10)
    for index,datum in enumerate(split_data):
        dvalue=datum.std()
        dvalues.append(dvalue)

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(Mguassian)
    plt.subplot(2,1,2)
    plt.pcolormesh((np.abs(cir_data)).T)
    try:
        pos = plt.ginput(n=-1)
    except:
        print("用中键退出,不要点叉")
        return 0
    plt.close()
    """
    input_string=input("输入中心点数字（1个或多个）")
    strings=input_string.split( )
    """

    """根据标签长度延长数据段长度"""
    if (len(tag_content)>=3):
        data_length+=50*(len(tag_content)-2)

    leften_length=int(data_length/2)
    righten_length=data_length-leften_length
    datacount=0
    dis=0
    for (index,s) in enumerate(pos):
        position = int(s[0])
        start_pos=position-leften_length
        if (start_pos<=0):
            start_pos=0
        print([start_pos,position+righten_length])
        datacount+=1
        tag_withinfo=tag_content
        if (index==0):
            tag_withinfo=tag_content
        else:
            tag_withinfo="="+tag_content
        if writecsv:
            writeCIR2csv(cir_data[start_pos:position+righten_length],tag_withinfo,output_path,distance="{}".format(dis))
        dis+=1
    return datacount


"""do segmentation to extract the moving part"""
def partition_cir(cir_file, tag_content="", output_path="./training_data", data_length=120,mode_forcelengthen=False,show_charts=False,effective_threshold=0.003):
    """读数据"""
    cir_data = np.genfromtxt(cir_file, dtype=complex, delimiter=',')

    """作差"""
    cir_diff = np.abs(np.diff(cir_data, axis=0))

    """对作差的数据求和"""
    dcir_sum = np.sum(cir_diff, axis=1)

    """平滑数据曲线"""
    Mguassian=gaussian_filter1d(dcir_sum * dcir_sum, 4)

    if show_charts:
        npsignalplot(Mguassian)

    data=Mguassian
    """作滑动窗口，分别计算每个窗口的标准差"""
    split_data=slidingwindow(data,10)
    temp=[]
    position_list=[]
    valid_flag=0
    positive_count=0
    begin_point=0
    end_point=0
    negative_count=0
    positive_threshold=effective_threshold
    negative_threshold=0.03
    """连续大于threshold开始，连续小于threshold结束"""
    for index,datum in enumerate(split_data):
        dvalue=datum.std()
        if valid_flag==0:
            """目前数据是静止的"""
            if dvalue>=positive_threshold:
                if positive_count==0:
                    """这一段数据可能是有效的，先记下起始点"""
                    begin_point=index
                positive_count+=1
            else:
                positive_count=0
            if positive_count>=4:
                """确认数据有效"""
                valid_flag=1
        elif valid_flag==1:
            """目前数据是有效的"""
            if dvalue<negative_threshold:
                if negative_count==0:
                    """可能是无效的开始，记下结束点"""
                    end_point=index
                negative_count+=1
            else:
                negative_count=0
            if negative_count>=40 or index>=split_data.size:
                """确认数据无效"""
                if (end_point-begin_point)>20:
                    """判断一下不会过短，写入结果数组"""
                    position_list.append([begin_point,end_point])
                valid_flag=0
        """便于调试，把分割依据的数据存下"""
        temp.append(dvalue)
    print(position_list)
    if show_charts:
        """画出分割依据的图"""
        npsignalplot(np.asarray(temp))


    """write data segmentation to disk"""
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    """作为文件名的tag"""
    tag_main=tag_content

    """遍历所有tag相同的文件数量，防止文件名重复"""
    data_count=0
    for root, dirs, files in os.walk(output_path):
        for filename in files:
            """遍历目标目录下所有文件"""
            if (len(filename)>len(tag_main)):
                if (filename[:len(tag_main)]==tag_main):
                    """相同tag的数据数量"""
                    data_count+=1

    base_path=output_path+"/"+tag_main+" "


    if (len(position_list)==3):
        """判断分割数量是否符合预期"""
        for positions in position_list:
            if mode_forcelengthen:
                """强制增强模式：以每段数据起点为基准强制补全到min_length，可以应对b这种第二笔较不明显的字母"""
                if (positions[1]-positions[0]<=min_length):
                    positions[1]=positions[0]+min_length
            if (positions[1]-positions[0]>=min_length):
                file_path=base_path+str(data_count)+'.csv'
                data_count+=1
                f = open(file_path, 'w', encoding='utf-8', newline='')
                csv_writer = csv.writer(f)
                """根据坐标写入数据，适当延长"""
                csv_writer.writerows(cir_data[positions[0]:positions[1]])

                f.close()

def drawCIR(filename):
    cir_data = np.genfromtxt(filename, dtype=complex, delimiter=',')
    plt.pcolormesh((np.abs(cir_data)).T)
    plt.show()

def drawDCIR(filename):
    cir_data = np.genfromtxt(filename, dtype=complex, delimiter=',')
    cir_diff = np.abs(np.diff(np.abs(cir_data), axis=0))
    plt.pcolormesh((np.abs(cir_diff)).T)
    plt.show()

if __name__ == '__main__':
    for i in range(0,21):
        filename = '..\GSM_generation\\training_data\\Word/require/jxy/require_{}.csv'.format(i)
        filename = 'expdata/zqword/1/100.csv'
        #filename = 'testdata/pixel3/1.csv'
        cir_data = np.genfromtxt(filename, dtype=complex, delimiter=',')
        #short_cir_data=(np.sum(np.abs(cir_data).reshape(-1,11),axis=1)).reshape(-1,11)

        static_filename='expdata/dr/{19}.csv'
        static_cir_data = np.genfromtxt(filename, dtype=complex, delimiter=',')
        static_cir_avg=np.average(static_cir_data,axis=0)
        cir_removestatic=cir_data-static_cir_avg

        drawCIR(filename)
        #作差
        cir_diff = np.abs(np.diff(np.abs(cir_data), axis=0))

        #short_cir_diff=np.abs(np.diff(short_cir_data, axis=0))

        # im = plt.imshow((b).T, interpolation='bilinear', cmap=cm.bwr )
        #dplt.pcolormesh((np.abs(cir_data)).T)
        plt.pcolormesh((np.abs(cir_diff)).T)
        #plt.pcolormesh((np.abs(cir_removestatic)).T)

        #plt.savefig('{0}n'.format(i))
        plt.show()
        #partition_word_bypeak(filename,"N",static_threshold=0.02,show_charts=True,desired_peaks=3,writecsv=True)
        partition_cir_manually(filename)
        #partition_cir(filename,"A",data_length=2000,show_charts=True,effective_threshold=0.005)
        #partition_word_bypeak(filename,"A",static_threshold=0.02,show_charts=True,desired_peaks=3,writecsv=True)
