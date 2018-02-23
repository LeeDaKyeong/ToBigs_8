import librosa
from scipy import signal
import util

# input : AudioSegment, output : AudioSegment
def noise_systhesis(audio, noise):
    combine = audio.overlay(noise)
    return combine

def noise_systhesis_path(audio_path, noise_path):
    audio = util.call_audio_AudioSegment(audio_path)
    noise = util.call_audio_AudioSegment(noise_path)

    combine = audio.overlay(noise)
    return combine

# denoise, input : librosa
def denoise(y):
    D_noise = librosa.stft(y)
    D_denoise = signal.wiener(D_noise)
    y_denoised = librosa.istft(D_denoise)
    return y_denoised

def denoise_path(path):
    y = util.call_audio_librosa(path)
    D_noise = librosa.stft(y)
    D_denoise = signal.wiener(D_noise)
    y_denoised = librosa.istft(D_denoise)
    return y_denoised