import util
import os
import numpy as np
from sklearn import svm
import pickle
import denoising
import re
from sklearn.cluster import KMeans

# speaker recog train data 생성(svm)

def train_data(path, Y, index):
    print("start")
    li = list()

    for index1 in os.listdir(path):
        path2 = os.path.join(path, index1)  # /Users/apple/Desktop/audio_name/DK/DK_11.30
        for index2 in os.listdir(path2):
            path3 = os.path.join(path2, index2)  # /Users/apple/Desktop/audio_name/DK/DK_11.30/DK_sentence1
            for index3 in os.listdir(path3):
                path4 = os.path.join(path3,
                                     index3)  # /Users/apple/Desktop/audio_name/DK/DK_11.30/DK_sentence1/sentence1_1.wav

                y, sr = util.call_audio_librosa(path4)
                # y = denoising.denoise(y)
                li.append(util.MFCC_extract_reshape(y, y_len=100000))
                Y.append(index)
    _li = np.array(li)

    return (_li, Y)


# svm model load
def speaker_recog_model_load(path = './model/speaker_recog_svm.sav'):
    model = pickle.load(open(path, 'rb'))
    return model

# test
def speaker_recog(audio_path):
    speaker = {0: "다경",
               1: "혜진",
               2: "강열",
               3: "이삭",
               4: "태권"}

    y,sr = util.call_audio_librosa(audio_path)
    if len(y) >= 15000:
        #print(len(y))
        mfcc = util.MFCC_extract_reshape(y, y_len = 100000)
        _mfcc = np.reshape(mfcc, (1, len(mfcc)))
        model = speaker_recog_model_load()
        result = model.predict(_mfcc)
        #print(speaker[int(result)])
        return speaker[int(result)]


# 시연할 때 util.word_seperation로 자른 단어들 이거로 화자인식
def speaker_recog_Aduio(AudioSeg):
    speaker = {0: "다경",
               1: "혜진",
               2: "강열",
               3: "이삭",
               4: "태권"}

    y = util.AudioSegment2librosa(AudioSeg)
    if len(y) >= 5000:
        mfcc = util.MFCC_extract_reshape(y, y_len=100000)
        _mfcc = np.reshape(mfcc, (1, len(mfcc)))
        model = speaker_recog_model_load()
        result = model.predict(_mfcc)

        #print(speaker[int(result)])
        return speaker[int(result)]

def speaker_clust(path):

    li = os.listdir(path)
    li = sorted(li, key = lambda x: (int(re.sub('[^0-9]','',x)),x))

    X = np.zeros((1,12544))
    for i in li:
        y, sr = util.call_audio_librosa(os.path.join(path,i))
        mfcc = util.MFCC_extract_reshape(y, y_len=100000)
        _mfcc = np.reshape(mfcc, (1, len(mfcc)))
        X = np.vstack((X,_mfcc))

    X = X[1:]

    kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
    print(kmeans.labels_)
    return kmeans.labels_



if __name__ == "__main__":
    Y = list()

    DK_path = "./audio_name/DK"
    DK, Y = train_data(DK_path, Y, 0)
    # DK_len = len(DK)

    HJ_path = "./audio_name/HJ"
    HJ, Y = train_data(HJ_path, Y, 1)
    # HJ_len = len(HJ)

    KY_path = "./audio_name/KY"
    KY, Y = train_data(KY_path, Y, 2)
    # KY_len = len(KY)

    LS_path = "./audio_name/LS"
    LS, Y = train_data(LS_path, Y, 3)
    # LS_len = len(LS)

    TK_path = "./audio_name/TK"
    TK, Y = train_data(TK_path, Y, 4)
    # TK_len = len(TK)

    X = np.vstack((DK, HJ, KY, LS, TK))

    # Y = np.zeros(DK_len+HJ_len+KY_len+LS_len+TK_len)
    # Y[:DK_len] = 0
    # Y[DK_len:HJ_len] = 1
    # Y[DK_len+HJ_len:DK_len+HJ_len+KY_len] = 2
    # Y[DK_len+HJ_len+KY_len:DK_len+HJ_len+KY_len+LS_len] = 3
    # Y[DK_len+HJ_len+KY_len+LS_len:] = 4

    lin_clf = svm.LinearSVC()
    lin_clf.fit(X, Y)

    filename = './model/speaker_recog_svm.sav'
    pickle.dump(lin_clf, open(filename, 'wb'))
