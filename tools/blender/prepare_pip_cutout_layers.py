from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'artifacts/cartoon/blender/pip_fullbody/pip_master_rgba.png'
OUT=ROOT/'artifacts/cartoon/blender/pip_cutout_layers'
OUT.mkdir(parents=True,exist_ok=True)
rgba=np.array(Image.open(SRC).convert('RGBA'))
rgb=rgba[:,:,:3]; alpha=rgba[:,:,3]
base=(alpha>70).astype(np.uint8)*255

def ellipse(center,axes,angle=0):
 m=np.zeros_like(base); cv2.ellipse(m,center,axes,angle,0,360,255,-1); return cv2.GaussianBlur(m,(0,0),1.0)

def poly(points):
 m=np.zeros_like(base); cv2.fillPoly(m,[np.array(points,np.int32)],255); return cv2.GaussianBlur(m,(0,0),1.0)

head=np.minimum(base,ellipse((575,815),(270,265)))
backpack=np.minimum(base,ellipse((250,1135),(175,335),-5))
left_arm=np.minimum(base,poly([(145,905),(255,850),(395,945),(500,1070),(430,1285),(260,1320),(150,1210)]))
right_arm=np.minimum(base,poly([(680,850),(820,840),(960,955),(940,1215),(790,1300),(690,1180)]))
core=np.maximum(ellipse((550,1235),(320,400)),poly([(345,1280),(750,1280),(750,1490),(700,1675),(500,1700),(300,1640)]))
torso=np.minimum(base,core)
for m in (head,backpack,left_arm,right_arm): torso=np.where(m>50,0,torso).astype(np.uint8)
parts={'torso':torso,'head':head,'left_arm':left_arm,'right_arm':right_arm,'backpack':backpack}
manifest={}
for name,m in parts.items():
 m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8)); m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8)); m=cv2.GaussianBlur(m,(0,0),0.55)
 Image.fromarray(np.dstack([rgb,m]),'RGBA').save(OUT/f'pip_{name}.png')
 x,y,w,h=cv2.boundingRect((m>128).astype(np.uint8)); manifest[name]=[int(x),int(y),int(w),int(h)]
# Neutralize the baked LED face so Blender's vector facial controls own the expression.
head_rgba=np.array(Image.open(OUT/'pip_head.png').convert('RGBA'))
screen_mask=ellipse((585,815),(205,150),0)
# solid sampled screen tone from a quiet patch of the original screen
screen_color=np.array([38,56,75],dtype=np.uint8)
inside=screen_mask>70
head_rgba[inside,0:3]=screen_color
head_rgba[inside,3]=np.maximum(head_rgba[inside,3],220)
Image.fromarray(head_rgba,'RGBA').save(OUT/'pip_head_neutral.png')
manifest['head_neutral']=manifest['head']
# clean plate from the trusted whole-character alpha
remove=cv2.dilate((base>50).astype(np.uint8)*255,np.ones((11,11),np.uint8)); bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR); plate=cv2.inpaint(bgr,remove,7,cv2.INPAINT_TELEA); plate=cv2.cvtColor(plate,cv2.COLOR_BGR2RGB)
Image.fromarray(plate,'RGB').save(OUT/'pip_background.jpg',quality=94)
(OUT/'manifest.json').write_text(json.dumps({'resolution':[1080,1920],'parts':manifest,'source':str(SRC),'facial_asset':'pip_head_neutral.png'},indent=2),encoding='utf-8')
print(json.dumps(manifest,indent=2))
