#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
from pydub import AudioSegment
import requests

path = "C:/soundfolder/"
audio_files = os.listdir('C:/soundfolder/')
# Folder is having audio files of both MP3 and WAV formats
len_audio=len(audio_files)
for i in range (len_audio):
    if os.path.splitext(audio_files[i])[1] == ".mp3":
        mp3_sound = AudioSegment.from_mp3(path+audio_files[i])
        print(mp3_sound)
        mp3_sound.export("<path>\\converted.wav", format="wav")