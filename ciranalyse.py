import csv
import os
from datetime import datetime
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from utility import npsignalplot,slidingwindow

filename = "gsmtest/fixed/8.csv"
filename = "gsmtest/ADIS/8.csv"


# filename="test.csv"

"""do segmentation to extract the moving part"""
def partition_cir(cir_file, tag_content="", output_path="./training_data"):
    """读数据"""
    cir_data = np.genfromtxt(cir_file, dtype=complex, delimiter=',')

    """作差"""
    cir_diff = np.abs(np.diff(cir_data, axis=0))

    dcir_sum = np.sum(cir_diff, axis=1)

    Mguassian=gaussian_filter1d(dcir_sum * dcir_sum, 4)

    npsignalplot(Mguassian)

    data=Mguassian
    split_data=slidingwindow(data,10)
    temp=[]
    position_list=[]
    valid_flag=0
    positive_count=0
    begin_point=0
    end_point=0
    negative_count=0
    positive_threshold=0.001
    negative_threshold=0.001
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
                if (end_point-begin_point)>25:
                    """判断一下不会过短，写入结果数组"""
                    position_list.append([begin_point,end_point])
                valid_flag=0
        temp.append(dvalue)
    print(position_list)
    npsignalplot(np.asarray(temp))


    """write data segmentation to disk"""
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    current_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    tag_main=tag_content

    data_count=0
    for root, dirs, files in os.walk(output_path):
        for filename in files:
            if (len(filename)>len(tag_main)):
                if (filename[:len(tag_main)]==tag_main):
                    data_count+=1

    base_path=output_path+"/"+tag_main+" "

    for positions in position_list:
        """筛选一定的长度"""
        if (positions[1]-positions[0]>=160):
            file_path=base_path+str(data_count)+'.csv'
            data_count+=1
            f = open(file_path, 'w', encoding='utf-8', newline='')
            csv_writer = csv.writer(f)
            csv_writer.writerows(cir_data[positions[0]:positions[1]])
            f.close()

if __name__ == '__main__':

    cir_data = np.genfromtxt(filename, dtype=complex, delimiter=',')

    #作差
    cir_diff = np.abs(np.diff(np.abs(cir_data), axis=0))



    # im = plt.imshow((b).T, interpolation='bilinear', cmap=cm.bwr )
    plt.pcolormesh((np.abs(cir_data)).T)
    plt.show()

    partition_cir(filename,"a")
