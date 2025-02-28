import librosa
import librosa.display
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.io.wavfile
from datasets import load_dataset
from huggingface_hub import login

token = os.getenv("HF_TOKEN")
login(token=token)

save_path = "mel_spectrograms"
os.makedirs(save_path, exist_ok=True)

dataset = load_dataset("Lagyamfi/akan_audio_processed")

def melspectrogram_feature(audio_array, sample_rate, filename, save_path):
    y = np.array(audio_array, dtype=np.float32)
#     sr, y = scipy.io.wavfile.read(audio_path)
#     if y.dtype!='float32':
#             y = y.astype('float32') / 32767.
    S = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_mels=64,n_fft=2048, hop_length=16,
                                       fmin=50,fmax=350)
    
    plt.figure(figsize=(2.25, 2.25))
    librosa.display.specshow(librosa.power_to_db(S,ref=np.max),
                            sr=sample_rate,
                            fmin=50,
                            fmax=350)
    
    plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
    plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
    plt.margins(0,0)

    save_name = filename + ".jpg"
    save_path = os.path.join(save_path, save_name)
    plt.savefig(save_path)
    plt.close('all')
    return save_path

for i, sample in enumerate(dataset["train"]):
    audio = sample["audio"]
    audio_array = audio["array"]
    sample_rate = audio["sampling_rate"]

    filename = f"sample_{i}"
    img_path = melspectrogram_feature(audio_array, sample_rate, filename, save_path)

    print(f"Saved: {img_path}")

#     name = os.path.basename(audio_path).split('.')[0] + '.jpg'
#     plt.savefig(os.path.join(save_path, name))
#     #plt.show()

#     plt.close('all')
#     return os.path.join(save_path, name)
   
