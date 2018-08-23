import numpy as np # for numerical operations
from moviepy.editor import VideoFileClip

clip = VideoFileClip(r"C:\Users\Room\Downloads\Matches\Highlights\liverpool-vs-west-ham\HL-liverpool-vs-west-ham_1.mp4")
cut = lambda i: clip.audio.subclip(i,i+1).to_soundarray(fps=22000)
volume = lambda array: np.sqrt(((1.0*array)**2).mean())
volumes = [volume(cut(i)) for i in range(0,int(clip.duration-1))]

frames = np.zeros((int(clip.duration)+1,clip.h,clip.w,3))
for i, frame in enumerate(clip.iter_frames(fps=1)):
    frames[0] = frame
a = 1
