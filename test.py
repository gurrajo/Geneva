from subprocess import check_output
import re


pts = str(check_output('ffmpeg -i video.mp4 -vf select="eq(pict_type\,I)" -an -vsync 0  keyframes%03d.jpg -loglevel debug 2>&1 |findstr select:1  ',shell=True),'utf-8')  #replace findstr with grep on Linux
pts = [float(i) for i in re.findall(r"\bpts:(\d+\.\d)", pts)] # Find pattern that starts with "pts:"
print(pts)

