import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import scipy.signal as signal
import csv

from datetime import datetime,timedelta
import numpy as np
import matplotlib.pyplot as plt
import json
import math
import numpy as np
from scipy.signal import stft
# from mid import base_dir

import math
import numpy as np
def MakeWindows(Name , n):
    n = n + 1 - n%2

    n = math.floor(n / 2)

    n = (np.linspace(start=1, stop=(2 * n + 1),num = 2 * n + 1) - (n + 1)) / 2 / n

    pu=-np.square(n) * 18

    if(Name == 'Gaussian'):
        win = np.exp(pu)
        win_return = win/np.linalg.norm(win)
    return win_return


def STFT(data, win, Hop_Size, Nfft, Fs):
    print("Calculating the STFT")
    data = signal.hilbert(np.real(data))
    L = len(win)
    Half = math.floor(L / 2)
    intdata_left = np.zeros((Half, 1))

    # print("intdata", intdata_right)
    data = np.append(intdata_left, data)
    cc = L - (len(data) % L)
    intdata_right = np.zeros((cc, 1))
    data = np.append(data, intdata_right)


    X = data.reshape(L, -1, order='F')

    for i in range(L):
        X[i, :] = X[i, :] * win[i]

    X = np.array(X, dtype=complex)
    # X =np.fft.fft2(X,s=[X.shape[1],12000])/math.sqrt(Nfft)
    # X = scipy.fft.fft2(X,s= [12000,11]) / math.sqrt(Nfft)
    X = np.fft.fft(X, 12000, axis=0) / math.sqrt(Nfft)

    X = abs(X[0:1000,1:X.shape[1] ])
    # print("X[1,:]",X[3,:]*win[0])
    # print("win[1,:]", win[1])
    # print("Over the STFT")
    print("Over the STFT")
    return X


def STFT2CALC(path):
    print("ok")
    my_matrix = np.loadtxt(open(path, "rb"), delimiter=",", skiprows=0)
    my_matrix = my_matrix - np.mean(my_matrix)

    for i in range(0, 1):
        # stft_specgram(my_matrix[i])
        # print("len(my_matrix)", len(my_matrix))
        Nfft = 2048
        win = MakeWindows('Gaussian', Nfft)

        zxxx = STFT(my_matrix, win, Nfft, 12000, 12000)

        # f, t, zxx = stft(my_matrix, window=win, fs=12000, nfft=12000, nperseg=2049)
        # f, t, zxx = stft(my_matrix, fs=12000, nfft=12000, nperseg=12000)
        # plt.pcolormesh(zxxx.shape[0], zxxx.shape[1], np.abs(zxxx), shading='auto', cmap='CMRmap')
        #############################################################################################
        #############################################################################################
        # time1=datetime.now()
        # plt.figure(1)
        # for i in range(zxxx.shape[1]):
        #     plt.plot(zxxx)
        # plt.savefig('./result/' + 'demo_stft_signl' + '.png')  # 保存图像
        # print("time1",datetime.now()-time1)
        #这部分的图像绘制需要大量时间，因此删除
        ##############################################################################################
        ##############################################################################################
        time2 = datetime.now()
        plt.figure(1)
        save_label = []
        save_label = abs(zxxx.T)
        b = np.ones(len(save_label))
        print(zxxx.shape)
        # 0: nLevel:(tt - 1) * nLevel
        cm = plt.cm.get_cmap('rainbow')
        plt.pcolormesh((Nfft / 12000) * np.linspace(start=0, stop=zxxx.shape[1], num=zxxx.shape[1]),
                       1000 * np.linspace(start=0, stop=1, num=zxxx.shape[0]), zxxx, shading='auto', cmap=cm)
        plt.axis([0, zxxx.shape[1] * Nfft / 12000, 0, 400])
        plt.colorbar(extend='both')
        print("time2",datetime.now()-time2)
        time3 = datetime.now()
        # save_label = np.insert(save_label, save_label.shape[1], b, axis=1)
        # print(save_label)
        save = pd.DataFrame(save_label)
        save.to_csv('./result/demo_stft.csv', index=False, header=False)
        print("time3",datetime.now()-time3)
        time4 = datetime.now()
        # plt.colorbar()
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.tight_layout()
        plt.legend()
    # plt.show()
    plt.savefig('./result/' + 'demo_stft' + '.png')  # 保存图像
    print("time4",datetime.now()-time4)



def STFTCALC(path):
    my_matrix = np.loadtxt(open(path,"rb"),delimiter=",",skiprows=0)
    plt.figure(2)
    for i in range(0, 1):
        # stft_specgram(my_matrix[i])
        # print("len(my_matrix)",len(my_matrix))
        win = MakeWindows('Gaussian',2048)
        f, t, zxx = stft(my_matrix, window=win,fs=12000,nfft=12000, nperseg=2049)
        # f, t, zxx = stft(my_matrix, fs=12000, nfft=12000, nperseg=12000)
        plt.pcolormesh(t, f, np.abs(zxx), shading='auto', cmap='CMRmap')
        save_label=[]
        save_label = abs(zxx)
        b = np.ones(len(save_label))
        save_label = np.insert(save_label, save_label.shape[1], b, axis=1)
        # print(save_label)
        save = pd.DataFrame(save_label)
        # save.to_csv(base_dir + './result/demo_stft.csv', index=False, header=False)
        plt.colorbar()
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.tight_layout()
        plt.legend()
    plt.savefig('./result/' + 'demo_stft' + '.png')  # 保存图像

def loaddata(file_name):
    my_matrix = np.loadtxt(open(file_name, "rb"), delimiter=",", skiprows=0)
    plt.figure(3)
    plt.plot(my_matrix)
    plt.savefig('./result/demo_signal.png', bbox_inches='tight')
    STFT2CALC(file_name)
    plt.close()#如果不close会一直覆盖最后造成堆栈溢出
    print("ok")


def stft(x, **params):
    '''
    :param x: 输入信号
    :param params: {fs:采样频率；
                    window:窗。默认为汉明窗；
                    nperseg： 每个段的长度，默认为256，
                    noverlap:重叠的点数。指定值时需要满足COLA约束。默认是窗长的一半，
                    nfft：fft长度，
                    detrend：（str、function或False）指定如何去趋势，默认为Flase，不去趋势。
                    return_onesided：默认为True，返回单边谱。
                    boundary：默认在时间序列两端添加0
                    padded：是否对时间序列进行填充0（当长度不够的时候），
                    axis：可以不必关心这个参数}
    :return: f:采样频率数组；t:段时间数组；Zxx:STFT结果
    '''
    f, t, zxx = signal.stft(x, **params)
    return f, t, zxx

def stft_specgram(x, picname=None, **params):    #picname是给图像的名字，为了保存图像
    f, t, zxx = stft(x, **params)
    plt.pcolormesh(t, f, np.abs(zxx),shading='auto')
    plt.colorbar()
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.tight_layout()
    plt.savefig('D:/Code/Project/Code3-Vibration/Flask/result/' + 'demo_stft' + '.png')  # 保存图像
    return t, f, zxx
