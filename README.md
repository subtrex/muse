# MuSE: Music from Scene Extraction

![slide-1](https://github.com/subtrex/muse/assets/53677987/c54e5799-a966-4289-a6ee-b4632a5fa159)

MuSE is a custom music generation app for your short form videos based on scene analysis.

### Inspiration

MuSE was inspired by the idea of a tool that can harness the power of generative artificial intelligence and produce relevant music for our short form videos based on the videos’ scenes and ambience of the video.

### What it does

MuSE is an AI-powered app that can produce relevant music for your videos based on its scene and ambience. It automatically generates custom music for videos by analyzing scenes and selecting appropriate music types, instruments, and moods.

### How we built it

**Web App Development:** For developing our web app, we utilized Streamlit, an open-source Python framework to deliver interactive data apps. We used tools like file upload, text input, and browser session manager provided by Streamlit to build our app.

**Frame Extraction:** We are using OpenCV, a popular computer vision library, to process the video and extract frames from it. We are extracting 6 frames from the video for scene analysis, all placed at regular intervals.

**Scene Analysis:** For scene analysis we are leveraging MIT's PlacesCNN model, a high performing model for scene recognition and categorizing deep scene features. The model takes the extracted video frames as input, and gives the following outputs: ambience (indoor or outdoor), scene categories, and scene attributes.

**Prompt Creation:** After scene analysis, we use the detected scene categories and attributes to associate relevant music type, instruments that should be present in the music piece, and how the audience is supposed to feel after hearing the music, or audience impression. We are using a clustering logic for the association, and producing a prompt based on the final chosen music type, instruments, and impression.

**Music Generation:** The prompt is passed to Meta's MusicGen model, a text to music model, for music generation. We are using the pre-trained musicgen-small model for generating the audio.

### Challenges we ran into

We faced challenges integrating an association algorithm that can effectively find the relevant music type, instrument, and audience impression based on the scene categories and attributes. We also faced challenges deploying the demo app on platforms like Streamlit Cloud, Heroku, and PythonAnywhere due to its memory intensive operations in the backend.

### Accomplishments that we're proud of

We successfully integrated the working between the various components of the app, namely the frame extraction, scene analysis, prompt creation and music generation, and got the general workflow to work, which was edifying.

### What we learned

As computer science graduates, we learnt a lot and gained deep insights into the concepts of scene recognition and music generation. We also learned how to integrate complex AI models into a cohesive application.

### What's next for MuSE

We can work on a more advanced clustering algorithm for better association of musical attributes, which in turn will produce better prompts. We can leverage the power of natural language processing to produce efficient prompts. We can use a more robust music generation model for better high quality audio.

### Demo

https://youtu.be/btImkH1TL5I
