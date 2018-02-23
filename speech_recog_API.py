import speech_recognition as sr

def speech_recog_google(path):
    # Generate Recognizer
    r = sr.Recognizer()

    with sr.AudioFile(path) as source:
        # record audio file object
        audio = r.record(source)
        try:
            result = (r.recognize_google(audio, language='ko'))
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))

    return result


# with microphone
def speech_recog_mic():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something!")
        audio = r.listen(source)

        try:
            result = (r.recognize_google(audio, language='ko'))
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))

    return result