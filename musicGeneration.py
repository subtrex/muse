from audiocraft.models import MusicGen
import streamlit as st 
import torch 
import torchaudio
import os 
import numpy as np
import base64
import io

@st.cache_resource
def load_model():
    model = MusicGen.get_pretrained('facebook/musicgen-small')
    return model

def generate_music_tensors(description, duration: int):
    model = load_model()

    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        duration=duration
    )

    output = model.generate(
        descriptions=[description],
        progress=True,
        return_tokens=True
    )

    return output[0]


def tensor2audio(samples: torch.Tensor):
    
    sample_rate = 32000
    assert samples.dim() == 2 or samples.dim() == 3

    samples = samples.detach().cpu()
    if samples.dim() == 2:
        samples = samples[None, ...]

    audio_files = []

    for idx, audio in enumerate(samples):
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio, sample_rate, format="wav")
        buffer.seek(0)
        audio_files.append(buffer)

    return audio_files

def getMusic(prompt, duration):
    music_tensors = generate_music_tensors(prompt, duration)
    final = tensor2audio(music_tensors)
    return final[0]


    


