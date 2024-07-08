import streamlit as st
from streamlit_lottie import st_lottie
import tempfile
import cv2
import os
import numpy as np
from io import BytesIO
from PIL import Image
from sceneRecognition import sceneRecogFunc
from promptGeneration import getPrompt
from musicGeneration import getMusic
import moviepy.editor as mp
import io
from collections import Counter

def combine_video_audio(video_file, audio_buffer):
    
    video_clip = mp.VideoFileClip(video_file)

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio_file:
        temp_audio_file.write(audio_buffer.read())

    audio_clip = mp.AudioFileClip(temp_audio_file.name)
    video_with_audio = video_clip.set_audio(audio_clip)

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video_file:
        temp_video_path = temp_video_file.name

    video_with_audio.write_videofile(temp_video_path, codec='libx264', audio_codec='aac', audio_bitrate='192k', fps=24, threads=4)

    with open(temp_video_path, 'rb') as f:
        final_buffer = io.BytesIO(f.read())

    final_buffer.seek(0)
    
    return final_buffer

st.set_page_config(page_title="MuSE", page_icon=":notes:", layout="wide")

with st.container():
    st.title("MuSE 🎵")
    st.markdown('<span style="font-size:20px; font-style: italic;"> Music from Scene Extraction </span>', unsafe_allow_html=True)
    st.write("---")
    st.write("Welcome to MuSE, a place to generate music for your videos based on scenes. MuSE utilizes the MIT's PlacesCNN model for scene detection, providing us with scene categories and attributes, on top of which we apply a clustering logic to produce an appropriate text prompt. We are then using Facebook's MusicGen model to generate music on the prompt for the video.")

with st.container():
    st.write("---")
    st.subheader("How to Use")

with st.container():
    col_1, col_2, col_3, col_4 = st.columns([1, 1, 1, 1], gap="large")

    with col_1:
        st.write("Upload your video")
        st.image('https://miro.medium.com/v2/resize:fit:1000/0*jxUH3Cwd-jlCnf3d', width=250)


    with col_2:
        st.write("Generate the music")
        st.image('https://www.devopsistech.com/wp-content/uploads/2022/09/App-development.gif', width=250)

    with col_3:
        st.write("View the result")    
        st.image('https://www.thoughtwin.com/assets/img/mernstack-img.gif', width=250)
    
    with col_4:
        st.write("Try with different video")
        st.image('https://gomycode.com/wp-content/uploads/2023/09/39998-web-development.gif', width=250)

with st.container():
    st.write("---")
    st.subheader("Dive In")

with st.container():
    st.markdown('<span style="font-size:20px; font-style: italic;"> Upload your video </span>', unsafe_allow_html=True)
    st.markdown('<span style="font-size:15px;"> For optimal results, please upload videos shorter than 15 seconds and with a minimum quality of 720p. </span>', unsafe_allow_html=True)

    video = st.file_uploader("Upload", label_visibility="collapsed", type=['mp4'])
    if(video is not None):
        readVideo = video.read()
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(readVideo)
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = int(total_frames/fps)
        interval = total_frames // 6
        frame_indices = [i * interval for i in range(6)]
        current_frame = 0
        extracted_frame_count = 0
        extracted_frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if current_frame in frame_indices:

                is_success, im_buf_arr = cv2.imencode(".jpg", frame)
                if is_success:
                    im_bytes = im_buf_arr.tobytes()
                    extracted_frames.append(im_bytes)
            current_frame += 1
        cap.release()

with st.container():
    st.write("---")
    st.markdown('<span style="font-size:20px; font-style: italic;"> Results </span>', unsafe_allow_html=True)
    if video is not None:
        st.markdown('<span style="font-size:20px;"> Scene Information </span>', unsafe_allow_html=True)

with st.container():
     if video is None:
        st.write("No video uploaded")
     else:   
        imgCol = st.columns(2)
        promptList = []
        for i, frame_bytes in enumerate(extracted_frames):
            image = Image.open(BytesIO(frame_bytes))
            col = imgCol[i % 2]  # Alternate between columns
            with col:
                innerCol_1, innerCol_2 = st.columns(2)
                with innerCol_1:
                    st.image(image, caption=f"Frame {i + 1}", width=250)
                with innerCol_2:
                    val = sceneRecogFunc(image)
                    prompt = getPrompt(val)
                    st.write("Ambience:", val[0])
                    st.write("Categories:", val[1])
                    st.write("Attributes:")
                    strAttribute = [str(x) for sublist in val[2] for x in sublist]
                    attrVal = ", ".join(strAttribute)
                    st.write(attrVal)
                    promptList.append(prompt)

with st.container():
    if video is not None:
        st.write("---")
        st.markdown('<span style="font-size:20px;"> Music Generation </span>', unsafe_allow_html=True)

if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def clickButton():
    st.session_state.clicked = True

with st.container():
    res_col_1, res_col_2, res_col_3 = st.columns([1, 3, 1], gap="large")

    with res_col_1:
        if video is None:
            st.write("")
        else:
            st.markdown('<span style="font-size:15px; font-weight: bold;"> Original Video </span>', unsafe_allow_html=True)
            st.video(readVideo)
                
    with res_col_2:
        if video is None:
            st.write("")
        else:
            st.markdown('<span style="font-size:15px; font-weight: bold;"> Generative Results </span>', unsafe_allow_html=True)

            st.write("Suggested prompts after scene analysis:")
            uniquePrompts = list(set(promptList))
            for x in range(0,len(uniquePrompts)):
                st.write((x+1),": ",uniquePrompts[x])
            txt = st.text_area("Enter the prompt number (e.g. 1 or 2)")

            if(not st.session_state.clicked):
                st.button("Generate", on_click=clickButton)

            if(st.session_state.clicked):

                with st.spinner('Generating. This may take a few minutes ...'):
                    #mostCommonPrompt = (lambda lst: Counter(lst).most_common(1)[0][0])(promptList)
                    genMusic = getMusic(uniquePrompts[int(txt)-1], duration+1)

                    st.write("Generated Music")
                    st.write("Prompt:", uniquePrompts[int(txt)-1])
                    st.audio(genMusic)

                    with res_col_3:
                        if video is None:
                            st.write("")
                        else:
                            st.markdown('<span style="font-size:15px; font-weight: bold;"> Video with Generated Music </span>', unsafe_allow_html=True)

                            stackVideo = combine_video_audio(tfile.name, genMusic)
                            st.video(stackVideo)
            else:
                st.write("")
    
    
            
                        
                 