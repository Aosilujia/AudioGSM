import os,shutil
import queue
from cirProcess import partition_cir,partition_word_bypeak,partition_cir_manually

MODE_VERIFY=True

def cir_cut_dir(path='.',output_path="./training_data"):
    """"""
    if not os.path.isdir(path):
        #file,not dir
        print(path+" is not a directory")
        return

    dirqueue = queue.Queue()
    dirqueue.put(path)
    while not dirqueue.empty():
        node = dirqueue.get()
        for filename in os.listdir(node):
            nextpath = os.path.join(node, filename)
            if os.path.isdir(nextpath):
                dirqueue.put(nextpath)
            elif nextpath.endswith('.csv'):
                content_tag=os.path.basename(node)
                user_tag=os.path.basename(os.path.split(node)[0])
                full_output_path=output_path+"/"+content_tag+"/"+user_tag
                print(nextpath)
                print(full_output_path)
                partition_word_bypeak(nextpath,tag_content=content_tag,output_path=full_output_path,
                                static_threshold=0.08,data_length=200,show_charts=MODE_VERIFY,writecsv=(not MODE_VERIFY),
                                desired_peaks=3)

def cir_cut_selected_files(filelist='./datalist.txt',output_path="./training_data"):
    if not os.path.exists(filelist):
        print("no such file named",filelist)
        return
    datalistfile=open(filelist,"r")
    datalist=datalistfile.readlines()
    dirqueue = queue.Queue()
    for datapath in datalist:
        datapath=datapath.strip('\n')
        if not os.path.exists(datapath):
            print("no such file named",datapath)
            continue;
        print(datapath)
        if os.path.isdir(datapath):
            dirqueue.put(datapath)
        elif datapath.endswith('.csv'):
            content_tag=os.path.basename(os.path.split(datapath)[0])
            user_tag=os.path.basename(os.path.split(os.path.split(datapath)[0])[0])
            full_output_path=output_path+"/"+content_tag+"/"+user_tag
            print(full_output_path)
            result=partition_cir_manually(datapath,tag_content=content_tag,output_path=full_output_path,
                                writecsv=True,data_length=200)
            if result!=0:
                mymovefile(datapath,"./processed_data/"+datapath)
            else:
                print("no data cut:",datapath)
    while not dirqueue.empty():
        node = dirqueue.get()
        for filename in os.listdir(node):
            nextpath = os.path.join(node, filename)
            if os.path.isdir(nextpath):
                dirqueue.put(nextpath)
            elif nextpath.endswith('.csv'):
                content_tag=os.path.basename(node)
                user_tag=os.path.basename(os.path.split(node)[0])
                full_output_path=output_path+"/"+content_tag+"/"+user_tag
                print(nextpath)
                result=partition_cir_manually(nextpath,tag_content=content_tag,output_path=full_output_path,
                                writecsv=True,data_length=200)
                if result!=0:
                    mymovefile(nextpath,"./processed_data/"+nextpath)
                else:
                    print("no data cut:",nextpath)


def mymovefile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.move(srcfile,dstfile)          #移动文件
        print("move %s -> %s"%( srcfile,dstfile))

MODE_VERIFY=False
#cir_cut_dir("expdata/jxyword/jxy/like")
cir_cut_selected_files()
