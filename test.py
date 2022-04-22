
import os
import re

#os.system('ffmpeg -hide_banner -i graphics\model_test4.mp4 -filter:v showinfo -y > graphics\info.txt 2>&1 output%d.png')

# Open a file: file
file = open('graphics\info.txt', mode='r')
t = [0]
for line in file:
    pts_P = re.findall(r"\spts_time:(\d\.\d+)", line) # Find pattern that starts with "pts:"
    if pts_P:
        t.append(float(pts_P[0]))
print(len(t))
# close the file
file.close()
