import os
import queue
from cirProcess import partition_cir,partition_word_bypeak

def cir_cut_dir(path='.',tag="",output_path="./training_data"):
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
                                static_threshold=0.02,data_length=200,show_charts=True,writecsv=False,
                                desired_peaks=3)

def cir_cut_selected_files(filelist='./datalist.txt',tag="",output_path="./training_data")


cir_cut_dir("expdata/jxy/O")
