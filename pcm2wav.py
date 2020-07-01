# -*- coding: UTF-8 -*-
# filename: pcm2wav date: 2018/11/22 12:54
# author: FD
import os
import queue
import wave


def pcm2wav(dirname):
    if dirname.endswith('.wav'):
        return
    if dirname.endswith('.pcm'):
        conv2wav(dirname)
        return
    filequeue = queue.Queue()
    filequeue.put(dirname)
    while not filequeue.empty():
        node = filequeue.get()
        for filename in os.listdir(node):
            nextpath = os.path.join(node, filename)
            if os.path.isdir(nextpath):
                filequeue.put(nextpath)
            elif nextpath.endswith('.pcm'):
                conv2wav(nextpath)

def conv2wav(nextpath):
    filenamenosuffix = nextpath[0: nextpath.index('.pcm')]
    wavfilepath = filenamenosuffix + ".wav"
    with open(nextpath, 'rb') as pcmfile:
        pcmdata = pcmfile.read()
    with wave.open(wavfilepath, 'wb') as wavfile:
        wavfile.setparams((1, 2, 48000, 0, 'NONE', 'NONE'))
        wavfile.writeframes(pcmdata)

if __name__ == '__main__':
    pcm2wav('.')
