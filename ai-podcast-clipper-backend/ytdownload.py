from pytubefix import YouTube
from pytubefix.cli import on_progress
import os

url = "https://www.youtube.com/watch?v=SOG0GmKts_I"
url2 = "https://www.youtube.com/live/-fs1SmF1cJA?si=zLblivwcXkMf-FNl"

yt = YouTube(url2, on_progress_callback=on_progress)
print(yt.title)

# ys = yt.streams.get_highest_resolution()
subtitles = yt.captions['a.en']
subtitles.save_captions('captions1.txt')
# ys.download()

# video_stream = yt.streams.filter(
#     res="1080p", file_extension="mp4", only_video=True).first()
# audio_stream = yt.streams.filter(
#     only_audio=True, file_extension="mp4").order_by('abr').desc().first()

# video_path = video_stream.download(filename="video.mp4")
# audio_path = audio_stream.download(filename="audio.mp4")

# output_path = "output_1080p.mp4"
# os.system(
#     f'ffmpeg -y -i "{video_path}" -i "{audio_path}" -c:v copy -c:a aac "{output_path}"')

# print(f"Downloaded and merged 1080p video to {output_path}")
