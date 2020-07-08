# audioCIR.py 开发者文档 #
## 概览 ##
gsm.py主要做两件事:  
task1:一件是基于gsm序列生成声波信号S（此处可参考论文 Strata Fine-Grained Acoustic-based Device-Free Tracking和 Endophasia_Utilizing Acoustic-Based Imaging for Issuing Contact-Free Silent Speech Commands）
；  
task2:一件是从使用S做实验收集到的信号R，从R上提取CIR（参考Channel Estimation Modeling）  
如何使用请参考README.md
## 代码结构 ##
task1的主函数为generate(),task2的主函数为estimate()，外层的各种工具函数都是调用这两个函数去完成任务。main函数的功能主要是处理命令行参数，提供用户接口。  
#### generate函数 ####
generate函数的主要流程如下：  
设置原始序列；  
调用umsample()将原始序列上取样到声波的频段（一般是人耳无法听见的20khz以上）；  
重复上取样的声音段，延长到所需的音频时长；  
最后将所有数据写到wav文件中；  
__上取样的upsample函数包含整个复杂的上取样过程，流水线如下：__  
将原始序列的0置为-1，得到一个只有1，-1的序列；  
在序列的末尾加上0序列，作为每帧之间的间隔；  
使用加零（zero-appending）延长序列，即先做fft，在fft结果中间加上一定数量的0，再ifft得到延长的信号。这样只改变了频率分布，又不影响大致的信号形状；  
把得到的信号作为numpy数组存到硬盘，以便之后estimate调用；  
对延长的信号乘上一个目标频率的正弦波，这样就把低频的原始序列变成了高频的声音信号；  
过一个带通滤波器，滤除无用频率的噪声；  
加上汉明窗，使信号平滑；  
#### estimate函数 ####
estimate函数的主要流程如下：  
打开wav文件，按照格式读入数据；  
去除手机录音产生的头部0段；  
调用downsample()对信号作下去样；  
使用framedetection()检测每一帧的起始位置，并且返回最早一帧的开头位置；  
根据得到的位置把所有数据分割成帧；  
对每一帧作处理提取CIR；  
写入csv文件；  
__下去样的downsample()函数流程如下：__  
将原始信号乘以一个余弦波作为实部，原始信号乘以正弦波作为虚部，两者分别过带通滤波器；  
__framedetection()函数流程如下：__  
读取generate()过程中存储的信号样本；  
取完整信号的一部分（因为这个处理过程速度较慢），与信号样本作cross-pearson-correlation；  
分析pearson的结果，选取帧位置；  

## ##
