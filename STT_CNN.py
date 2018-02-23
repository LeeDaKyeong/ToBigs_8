import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import re

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import load_model


def train_image_generate(input_path, output_path):
    path = input_path
    words = ['word1', 'word2', 'word3', 'word4', 'word5', 'word6', 'word7', 'word8', 'word9', 'word10', 'word11', 'word12']
    for word in words:
        #print(word)
        path_word = os.path.join(path, word)

        if ".DS" in path_word:
            continue
        else:
            for audio in os.listdir(path_word):
                y, sr = librosa.load(os.path.join(path_word, audio), sr=44100)

                fig = plt.figure(figsize=(5, 3))
                MFCC = librosa.feature.mfcc(y=y, sr=sr)
                librosa.display.specshow(MFCC, x_axis='time', y_axis='mel')

                save_dir = output_path + word + "/" + audio.split(".")[0] + ".png"
                # print(save_dir)
                fig.savefig(save_dir)  # save the figure to file
                plt.close(fig)
    return

def batch_generating(path):
    path = path

    datagen = ImageDataGenerator(rescale=1. / 255)

    batch_generator = datagen.flow_from_directory(
        path,
        # color_mode="rgb",
        target_size=(216, 360),
        batch_size=32,
        class_mode='categorical')

    return batch_generator

def modeling():
    model = Sequential()
    model.add(
        Conv2D(16, kernel_size=(3, 3), activation='elu', input_shape=(216, 360, 3), kernel_initializer='glorot_normal'))

    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=1, activation='elu', kernel_initializer='glorot_normal'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(64, (3, 3), activation='elu', strides=1, kernel_initializer='glorot_normal'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(128, (3, 3), activation='elu', strides=1, kernel_initializer='glorot_normal'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Flatten())
    model.add(Dense(64, activation='elu', kernel_initializer='glorot_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='elu', kernel_initializer='glorot_normal'))

    model.add(Dense(len(train_generator.class_indices), activation='softmax'))
    #model.summary()

    return model

def test_image_generating():
    li = os.listdir("./test")
    # print(li)
    li = sorted(li, key=lambda x: (int(re.sub('[^0-9]', '', x)), x))

    for i in li:
        path = "/Users/apple/Desktop/chunk_image/" + i.split(".")[0]
        os.makedirs(path, exist_ok=True)

        _path = os.path.join(path,"chunk")
        os.makedirs(_path, exist_ok=True)

        y, sr = librosa.load(os.path.join("./test", i), sr=44100)

        fig = plt.figure(figsize=(5, 3))
        MFCC = librosa.feature.mfcc(y=y, sr=sr)
        librosa.display.specshow(MFCC, x_axis='time', y_axis='mel')

        save_dir = _path + "/" + i.split(".")[0] + ".png"
        # print(save_dir)
        fig.savefig(save_dir)  # save the figure to file
        plt.close(fig)


def model_laoad(path):
    model = load_model(path)
    return model

def speech_recognition(path, model1, model2):
    speech = {0: "안녕하세요",
              1: "지금",
              2: "들어주셔서",
              3: "감사합니다",
              4: "빅보이스",
              5: "팀",
              6: "입니다",
              7: "발표는",
              8: "어떠셨나요",
              9: "저희",
              10: "여기",
              11: "까지"}

    #model1 = load_model('./model/STT_CNN1_mfcc.h5')
    #model2 = load_model('./model/STT_CNN1_mfcc2.h5')

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        path,
        target_size=(216, 360),
        batch_size=1,
        class_mode='categorical',
        shuffle=False)

    prob1 = model1.predict_generator(test_generator)
    prob2 = model2.predict_generator(test_generator)

    prob = prob1+prob2
    #print(speech[np.argmax(prob)])
    return speech[np.argmax(prob)]



if __name__ == "__main__":
    train_image_generate("./train_audio","./train_image/")
    train_image_generate("./test_audio", "./test_image/")

    train_generator = batch_generating('./train_image')
    test_generator = batch_generating('./test_image')

    model1 = modeling()
    model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    early = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, verbose=1, mode='auto')
    # early = val 로스가 0에 수렴하거나, 5번 멈춰있으면 train 종료
    reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, mode='auto', min_lr=0.0000001)
    # reduce  = val 로스가 진전이 없는 경우, learning_rate = 5번 멈춰있으면 train 종료

    hist1 = model1.fit_generator(
        train_generator,
        steps_per_epoch=100,  # 32개 이미지를 100번 학습시키는 것이 1 epoch로 정의.
        epochs=10,
        validation_data=test_generator,
        validation_steps=10)
    # callbacks=[early, reduce])

    model1.save('./model/STT_CNN1_mfcc.h5')

    model2 = modeling()
    model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    hist2 = model2.fit_generator(
        train_generator,
        steps_per_epoch=100,  # 32개 이미지를 100번 학습시키는 것이 1 epoch로 정의.
        epochs=10,
        validation_data=test_generator,
        validation_steps=10)
    # callbacks=[early, reduce])

    model2.save('./model/STT_CNN1_mfcc2.h5')