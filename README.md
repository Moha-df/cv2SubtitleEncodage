
# créer un venv
python -m venv venv  

# activer
source venv/bin/activate


# installer les deps
pip install -r requirements.txt

# Encoder la video.mp4
python .\encode_subtitles_visible_large.py --video video.mp4 --srt subtitle.srt --output videoTest2.mp4 --point-size 12 --camouflage 90 --local-radius 50 --alea 50