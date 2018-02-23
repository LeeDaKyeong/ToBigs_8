import os
import re
import speaker_recog
import speech_recog_API
import STT_CNN

if __name__ == "__main__":

    #audio = util.call_audio_AudioSegment('testtest.wav')
    #chunk = util.word_seperation(audio)

    model1 = STT_CNN.model_laoad('./model/STT_CNN1_mfcc.h5')
    model2 = STT_CNN.model_laoad('./model/STT_CNN1_mfcc2.h5')

    li = os.listdir("./testtest")
    li = sorted(li, key = lambda x: (int(re.sub('[^0-9]','',x)),x))

    for i in li:
        path = os.path.join("./testtest",i)

        for j in os.listdir(path):
            if "wav" in j:
                speaker = speaker_recog.speaker_recog(os.path.join(path,j))
            else:
                speech = STT_CNN.speech_recognition(path, model1, model2)
        print("speaker : ",speaker, "speech : ",speech)



    #print(speech_recog_API.speech_recog_mic())



