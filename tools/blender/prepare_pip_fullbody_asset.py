from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'assets' / 'cartoon' / 'pip.png'
OUT = ROOT / 'artifacts' / 'cartoon' / 'blender' / 'pip_fullbody'
OUT.mkdir(parents=True, exist_ok=True)

im = Image.open(SRC).convert('RGB')
scale = max(1080 / im.width, 1920 / im.height)
im = im.resize((round(im.width * scale), round(im.height * scale)), Image.Resampling.LANCZOS)
x = max(0, (im.width - 1080) // 2); y = max(0, (im.height - 1920) // 2)
rgb = np.array(im.crop((x, y, x + 1080, y + 1920)))

hsv = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(hsv)
r,g,b = [rgb[:,:,i].astype(np.int16) for i in range(3)]
blue = ((b >= r + 18) & (b >= g + 2) & ((b-r)+(b-g) >= 22) & (S >= 28) & (V >= 75)).astype(np.uint8) * 255
orange = ((r >= b + 30) & (r >= g + 18) & (r > 125) & (S >= 55) & (V >= 75)).astype(np.uint8) * 255
color = np.maximum(blue, orange)

# Tight global silhouette window around PIP; prevents pastel scene furniture from
# becoming part of the deformable character texture.
poly = np.array([
    [250,560],[420,540],[600,555],[755,625],[845,760],[885,930],[950,1030],
    [960,1265],[880,1370],[790,1420],[710,1510],[680,1635],[520,1680],
    [390,1650],[315,1530],[220,1390],[160,1240],[165,1000],[195,820],
], np.int32)
sil = np.zeros((1920,1080), np.uint8); cv2.fillPoly(sil,[poly],255)
base_color = cv2.bitwise_and(color, sil)
base_color = cv2.morphologyEx(base_color, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8))

# Keep only the dominant connected character component and its two large feet.
num, labels, stats, centroids = cv2.connectedComponentsWithStats((base_color > 128).astype(np.uint8), 8)
areas = stats[:,4]
order = np.argsort(areas)[::-1]
keep = np.zeros_like(base_color)
kept=[]
for idx in order[1:]:
    area=int(areas[idx]); x0,y0,w,h,a=map(int,stats[idx])
    cx,cy=centroids[idx]
    central=(180 <= cx <= 900 and 520 <= cy <= 1700)
    if (not kept and area > 50000) or (central and area > 5000 and cy > 1250 and len(kept)<3):
        keep[labels==idx]=255; kept.append(int(idx))

# Add the dark LED face screen only inside a tight head ellipse.
head_geom=np.zeros_like(keep); cv2.ellipse(head_geom,(545,835),(285,275),0,0,360,255,-1)
dark=((V<150)&(S<110)).astype(np.uint8)*255
keep=np.maximum(keep,cv2.bitwise_and(dark,head_geom))
keep=cv2.morphologyEx(keep,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
keep=cv2.GaussianBlur(keep,(0,0),0.75)

# Full-body RGBA texture used by the Blender armature-weighted mesh.
Image.fromarray(np.dstack([rgb,keep]),'RGBA').save(OUT/'pip_master_rgba.png')

# Character-free background plate for free motion.
remove=cv2.dilate((keep>55).astype(np.uint8)*255,np.ones((13,13),np.uint8))
bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
plate=cv2.inpaint(bgr,remove,7,cv2.INPAINT_TELEA); plate=cv2.cvtColor(plate,cv2.COLOR_BGR2RGB)
Image.fromarray(plate,'RGB').save(OUT/'pip_background.jpg',quality=94)

ys,xs=np.where(keep>128)
manifest={
  'resolution':[1080,1920],
  'bbox':[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)],
  'coverage':round(float((keep>128).mean()),4),
  'components_kept':kept,
  'notes':'clean-master-fullbody-silhouette-for-blender-armature',
}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest,indent=2))
