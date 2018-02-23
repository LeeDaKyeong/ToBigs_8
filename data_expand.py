import librosa
import os
import random

def pitchUp(y, sr = 44100, n_step = 10):
    y_pitch_higher = librosa.effects.pitch_shift(y, sr, n_steps=n_step)
    return y_pitch_higher


def pitchDown(y, sr = 44100, n_step = -10):
    y_pitch_lower = librosa.effects.pitch_shift(y, sr, n_steps=n_step)
    return y_pitch_lower

def speedUp(y, n_step = 2):
    y_D = librosa.stft(y)
    y_D_fast = librosa.phase_vocoder(y_D, n_step)
    y_faster = librosa.istft(y_D_fast)
    return y_faster


def speedDown(y, n_step=0.5):
    y_D = librosa.stft(y)
    y_D_slow = librosa.phase_vocoder(y_D, n_step)
    y_slower = librosa.istft(y_D_slow)
    return y_slower

def train_test_split():
    train_path = "./train_audio"
    test_path = "./test_audio"

    for i in range(len(words)):
        print(i)
        train_word_path = os.path.join(train_path, words[i])  # "/Users/apple/Desktop/train_audio/word1"
        test_word_path = os.path.join(test_path, words[i])  # "/Users/apple/Desktop/test_audio/word1"

        length = len(os.listdir(train_word_path))
        test_index = random.sample(range(length), int(length * 0.3))
        test_index.sort(reverse=True)

        for j in test_index:
            before = os.path.join(train_word_path, os.listdir(train_word_path)[j])
            after = os.path.join(test_word_path, os.listdir(train_word_path)[j])

            os.rename(before, after)
    return


if __name__ == "__main__":
    path = "./audio_date/12.29"
    words = ['word1', 'word2', 'word3', 'word4', 'word5', 'word6', 'word7', 'word8', 'word9', 'word10', 'word11', 'word12']

    save_dir = "./trian_audio"
    for person in os.listdir(path):
        path_person = os.path.join(path, person)

        for word in words:
            path_word = os.path.join(path_person, word)

            for audio in os.listdir(path_word):
                path_audio = os.path.join(path_word, audio)

                y, sr = librosa.load(path_audio, sr=44100)
                y = y[1000:]

                y1 = pitchUp(y)
                y2 = pitchDown(y)
                y3 = speedUp(y)
                y4 = speedDown(y)

                save_path = os.path.join(save_dir, word)
                print(save_path)

                librosa.output.write_wav(os.path.join(save_path, audio), y=y, sr=44100)  # 저장
                librosa.output.write_wav(os.path.join(save_path, audio.split('.')[0] + "pitchUp.wav"), y=y1, sr=44100)  # 저장
                librosa.output.write_wav(os.path.join(save_path, audio.split('.')[0] + "pitchDown.wav"), y=y2, sr=44100)  # 저장
                librosa.output.write_wav(os.path.join(save_path, audio.split('.')[0] + "speedUp.wav"), y=y3, sr=44100)  # 저장
                librosa.output.write_wav(os.path.join(save_path, audio.split('.')[0] + "speedDown.wav"), y=y4, sr=44100)  # 저장


    #랜덤하게 트레인 테스트 나누기
    train_test_split()

