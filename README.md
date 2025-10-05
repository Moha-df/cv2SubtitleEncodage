
# créer un venv
python -m venv venv  

# activer
source venv/bin/activate


# installer les deps
pip install -r requirements.txt

# Encoder la video.mp4
python encode_subtitles_visible_large.py --video video.mp4 --srt subtitle.srt --output video_16x16_4grids_white_with_borders.mp4

# Decoder la video depuis le pc direct (ca reste un decodage visuelle)
python decode_camera.py video_16x16_4grids_white_with_borders.mp4

# Decoder la video avec une camera (remplacer 10 par le num de la cam)
python decode_camera.py 10 

# Lancer la video avec mpv
mpv video_16x16_4grids_white_with_borders.mp4