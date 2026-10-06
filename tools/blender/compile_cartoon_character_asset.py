from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "assets" / "cartoon" / "characters"
OUT_ROOT = ROOT / "artifacts" / "cartoon" / "blender"
W, H = 1080, 1920

# One stable hero pose is selected from each multi-pose character sheet. The rest
# of the sheet never enters the rig, which prevents duplicate bodies and labels.
SPECS = {
    "MOMO": {
        "source": SRC_DIR / "momo.png",
        "crop": (55, 300, 430, 780),
        "target_box": (230, 420, 850, 1700),
        "head_rel": (0.08, 0.02, 0.92, 0.46),
        "left_arm_rel": (0.00, 0.40, 0.42, 0.86),
        "right_arm_rel": (0.58, 0.40, 1.00, 0.86),
        "backpack_rel": (0.00, 0.80, 0.08, 0.98),
        "face_rel": (0.22, 0.22, 0.80, 0.48),
        "eyes": ((0.36, 0.28, 0.14, 0.10), (0.66, 0.28, 0.14, 0.10)),
        "mouth": (0.50, 0.38, 0.12, 0.045),
        "mouth_kind": "organic",
        "palette": [(0.82, 0.42, 0.12), (0.09, 0.34, 0.36)],
    },
    "BUBU": {
        "source": SRC_DIR / "bubu.png",
        "crop": (245, 265, 515, 610),
        "target_box": (250, 430, 830, 1700),
        "head_rel": (0.08, 0.02, 0.92, 0.50),
        "left_arm_rel": (0.00, 0.42, 0.37, 0.83),
        "right_arm_rel": (0.63, 0.42, 1.00, 0.83),
        "backpack_rel": (0.72, 0.48, 1.00, 0.84),
        "face_rel": (0.18, 0.20, 0.83, 0.57),
        "eyes": ((0.34, 0.29, 0.17, 0.13), (0.66, 0.29, 0.17, 0.13)),
        "mouth": (0.50, 0.40, 0.18, 0.065),
        "mouth_kind": "organic",
        "palette": [(0.10, 0.38, 0.70), (0.96, 0.68, 0.10)],
    },
}


def _rgba_crop(src: Image.Image, box: tuple[int,int,int,int], target_box: tuple[int,int,int,int]) -> tuple[np.ndarray, tuple[float,float]]:
    crop = src.crop(box).convert("RGB")
    cw, ch = crop.size
    out_w = target_box[2] - target_box[0]
    out_h = target_box[3] - target_box[1]
    scale = min(out_w / cw, out_h / ch)
    rw, rh = round(cw * scale), round(ch * scale)
    crop = crop.resize((rw, rh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (W, H), (245, 245, 245))
    x = target_box[0] + (out_w - rw) // 2
    y = target_box[1] + (out_h - rh) // 2
    canvas.paste(crop, (x, y))
    return np.asarray(canvas), (scale, scale)


def _grabcut(rgb: np.ndarray, target_box: tuple[int,int,int,int]) -> np.ndarray:
    mask = np.full((H, W), cv2.GC_BGD, np.uint8)
    x0, y0, x1, y1 = target_box
    ix = max(x0 + 18, x0); iy = max(y0 + 18, y0)
    iw = max(2, x1 - x0 - 36); ih = max(2, y1 - y0 - 36)
    mask[iy:iy+ih, ix:ix+iw] = cv2.GC_PR_FGD
    # GrabCut needs the standard 1x65 GMM state arrays when initialized from a mask.
    bgd = np.zeros((1, 65), np.float64); fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask, None, bgd, fgd, 6, cv2.GC_INIT_WITH_MASK)
    alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    r,g,b=[rgb[:,:,i] for i in range(3)]
    pale=(r>228)&(g>220)&(b>205)
    alpha[pale]=0
    # Preserve tiny paws/feet while killing fuzzy background halos.
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    alpha = cv2.GaussianBlur(alpha, (0,0), 0.7)
    return alpha


def _rel_box(rel, target_box):
    x0,y0,x1,y1=target_box
    return (round(x0+(x1-x0)*rel[0]), round(y0+(y1-y0)*rel[1]), round(x0+(x1-x0)*rel[2]), round(y0+(y1-y0)*rel[3]))


def _ellipse_mask(center_rel, axes_rel, target_box):
    x0,y0,x1,y1=target_box; cx=x0+(x1-x0)*center_rel[0]; cy=y0+(y1-y0)*center_rel[1]
    ax=(x1-x0)*axes_rel[0]/2; ay=(y1-y0)*axes_rel[1]/2
    m=np.zeros((H,W),np.uint8); cv2.ellipse(m,(round(cx),round(cy)),(max(1,round(ax)),max(1,round(ay))),0,0,360,255,-1); return m


def _layer(rgb, alpha, region):
    return Image.fromarray(np.dstack([rgb, np.minimum(alpha, region)]).astype(np.uint8),'RGBA')


def compile_character(name: str) -> dict:
    spec=SPECS[name]; src_path=spec['source']; out=OUT_ROOT/f'{name.lower()}_cutout_layers'; out.mkdir(parents=True,exist_ok=True)
    src=Image.open(src_path).convert('RGB'); crop_box=spec['crop']; target_box=spec['target_box']
    rgb,_=_rgba_crop(src,crop_box,target_box); alpha=_grabcut(rgb,target_box)
    # Keep connected component(s) inside target area; removes any stray background pickup.
    region=np.zeros((H,W),np.uint8); cv2.rectangle(region,(target_box[0],target_box[1]),(target_box[2],target_box[3]),255,-1)
    alpha=np.minimum(alpha,region)

    head=_ellipse_mask(((spec['head_rel'][0]+spec['head_rel'][2])/2,(spec['head_rel'][1]+spec['head_rel'][3])/2),(spec['head_rel'][2]-spec['head_rel'][0],spec['head_rel'][3]-spec['head_rel'][1]),target_box)
    left=_ellipse_mask(((spec['left_arm_rel'][0]+spec['left_arm_rel'][2])/2,(spec['left_arm_rel'][1]+spec['left_arm_rel'][3])/2),(spec['left_arm_rel'][2]-spec['left_arm_rel'][0],spec['left_arm_rel'][3]-spec['left_arm_rel'][1]),target_box)
    right=_ellipse_mask(((spec['right_arm_rel'][0]+spec['right_arm_rel'][2])/2,(spec['right_arm_rel'][1]+spec['right_arm_rel'][3])/2),(spec['right_arm_rel'][2]-spec['right_arm_rel'][0],spec['right_arm_rel'][3]-spec['right_arm_rel'][1]),target_box)
    pack=_ellipse_mask(((spec['backpack_rel'][0]+spec['backpack_rel'][2])/2,(spec['backpack_rel'][1]+spec['backpack_rel'][3])/2),(spec['backpack_rel'][2]-spec['backpack_rel'][0],spec['backpack_rel'][3]-spec['backpack_rel'][1]),target_box)
    head=np.minimum(head,alpha); left=np.minimum(left,alpha); right=np.minimum(right,alpha); pack=np.minimum(pack,alpha)
    torso=alpha.copy(); torso=cv2.morphologyEx(torso,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))

    # Organic face system: keep authentic eyes/beak/mouth as texture layers, while
    # leaving the head layer neutralized underneath for separate eye/mouth animation.
    face=_rel_box(spec['face_rel'],target_box)
    fmask=np.zeros((H,W),np.uint8); cv2.rectangle(fmask,(face[0],face[1]),(face[2],face[3]),255,-1); fmask=cv2.GaussianBlur(fmask,(0,0),2.0); fmask=np.minimum(fmask,head)
    def rel_feature(item):
        cx,cy,ww,hh=item; return (round(target_box[0]+(target_box[2]-target_box[0])*cx), round(target_box[1]+(target_box[3]-target_box[1])*cy), round((target_box[2]-target_box[0])*ww), round((target_box[3]-target_box[1])*hh))
    eye_l=rel_feature(spec['eyes'][0]); eye_r=rel_feature(spec['eyes'][1]); mouth_c=rel_feature(spec['mouth'])
    def ellipse_feature(item):
        cx,cy,w,h=item; m=np.zeros((H,W),np.uint8); cv2.ellipse(m,(cx,cy),(max(2,w//2),max(2,h//2)),0,0,360,255,-1); return np.minimum(m,fmask)
    eye_lm=ellipse_feature(eye_l); eye_rm=ellipse_feature(eye_r); mouth_m=ellipse_feature(mouth_c)
    face_parts=cv2.bitwise_or(cv2.bitwise_or(eye_lm,eye_rm),mouth_m)
    # Authentic organic face stays on the neutral head; the eyes/mouth become
    # independent texture patches so they can move/scale with the performance.
    bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR); neutral_rgb=cv2.cvtColor(cv2.inpaint(bgr,(face_parts>35).astype(np.uint8)*255,5,cv2.INPAINT_TELEA),cv2.COLOR_BGR2RGB)
    head_neutral=np.minimum(head,np.where(face_parts>35,0,255).astype(np.uint8))
    face_base=np.minimum(fmask,head); head_neutral=np.maximum(head_neutral,np.where((face_base>35)&(face_parts<=35),face_base,0).astype(np.uint8))
    # MOMO's source head already has a high-quality organic facial expression.
    # Keep that face intact for the production rig instead of cutting rectangular
    # eye/mouth holes and re-compositing small patches over the fur.
    head_out = head if name == 'MOMO' else head_neutral
    Image.fromarray(np.dstack([rgb,head_out]),'RGBA').save(out/f'{name.lower()}_head.png')
    Image.fromarray(np.dstack([rgb,head_out]),'RGBA').save(out/f'{name.lower()}_head_neutral.png')
    Image.fromarray(np.dstack([rgb,eye_lm]),'RGBA').save(out/f'{name.lower()}_eye_left.png')
    Image.fromarray(np.dstack([rgb,eye_rm]),'RGBA').save(out/f'{name.lower()}_eye_right.png')
    Image.fromarray(np.dstack([rgb,mouth_m]),'RGBA').save(out/f'{name.lower()}_mouth.png')
    Image.fromarray(np.dstack([rgb,torso]),'RGBA').save(out/f'{name.lower()}_torso.png')
    Image.fromarray(np.dstack([rgb,left]),'RGBA').save(out/f'{name.lower()}_left_arm.png')
    Image.fromarray(np.dstack([rgb,right]),'RGBA').save(out/f'{name.lower()}_right_arm.png')
    accessory_name='scarf' if name=='MOMO' else 'backpack'
    Image.fromarray(np.dstack([rgb,pack]),'RGBA').save(out/f'{name.lower()}_{accessory_name}.png')
    # Full authentic face patch is retained as a source reference.
    Image.fromarray(np.dstack([rgb,fmask]),'RGBA').save(out/f'{name.lower()}_face.png')

    face_rel=spec['face_rel']; fx0,fy0,fx1,fy1=face
    ex0=int(target_box[0]+(target_box[2]-target_box[0])*spec['eyes'][0][0]); ey0=int(target_box[1]+(target_box[3]-target_box[1])*spec['eyes'][0][1])
    # Manifest is in absolute 1080x1920 coordinates; store compact profile values.
    center=lambda rel:(round(target_box[0]+(target_box[2]-target_box[0])*rel[0]),round(target_box[1]+(target_box[3]-target_box[1])*rel[1]))
    root=center((0.50,0.82)); head_center=center(((spec['head_rel'][0]+spec['head_rel'][2])/2,(spec['head_rel'][1]+spec['head_rel'][3])/2))
    shoulder_y=round(target_box[1]+(target_box[3]-target_box[1])*0.62); elbow_y=round(target_box[1]+(target_box[3]-target_box[1])*0.72); wrist_y=round(target_box[1]+(target_box[3]-target_box[1])*0.82)
    shoulder_l=center((0.32,0.60)); shoulder_r=center((0.68,0.60)); elbow_l=center((0.20,0.70)); elbow_r=center((0.80,0.70)); wrist_l=center((0.16,0.82)); wrist_r=center((0.84,0.82))
    manifest={
      'character':name,'resolution':[W,H],'source':str(src_path),'mode':'compiled-single-hero-pose','layers':{},
      'parts':{k:[int(v[0]),int(v[1]),int(v[2]),int(v[3])] for k,v in {
        'torso':cv2.boundingRect((torso>128).astype(np.uint8)),'head':cv2.boundingRect((head>128).astype(np.uint8)),'left_arm':cv2.boundingRect((left>128).astype(np.uint8)),'right_arm':cv2.boundingRect((right>128).astype(np.uint8)),'backpack':cv2.boundingRect((pack>128).astype(np.uint8))}.items()},
      'face':{'box':[fx0,fy0,fx1,fy1],'mode':spec['mouth_kind'],'eye_left':[eye_l[0]-eye_l[2]//2,eye_l[1]-eye_l[3]//2,eye_l[0]+eye_l[2]//2,eye_l[1]+eye_l[3]//2],'eye_right':[eye_r[0]-eye_r[2]//2,eye_r[1]-eye_r[3]//2,eye_r[0]+eye_r[2]//2,eye_r[1]+eye_r[3]//2],'mouth':[mouth_c[0]-mouth_c[2]//2,mouth_c[1]-mouth_c[3]//2,mouth_c[0]+mouth_c[2]//2,mouth_c[1]+mouth_c[3]//2]},
      'face_assets':{'face':f'{name.lower()}_face.png','head_neutral':f'{name.lower()}_head_neutral.png','eye_left':f'{name.lower()}_eye_left.png','eye_right':f'{name.lower()}_eye_right.png','mouth':f'{name.lower()}_mouth.png'},
      'rig_points':{'root':root,'head_center':head_center,'left_shoulder':shoulder_l,'left_elbow':elbow_l,'left_wrist':wrist_l,'right_shoulder':shoulder_r,'right_elbow':elbow_r,'right_wrist':wrist_r},
    }
    manifest['parts']['head_neutral']=manifest['parts']['head'];
    cv2.imwrite(str(out/f'{name.lower()}_background.jpg'), cv2.cvtColor(cv2.inpaint(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),(alpha>40).astype(np.uint8)*255,7,cv2.INPAINT_TELEA),cv2.COLOR_BGR2RGB))
    # Use fixed 1080x1920 paths in the Blender profile.
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('characters',nargs='*',default=['MOMO','BUBU']); args=ap.parse_args()
    names=[x.upper() for x in args.characters] or ['MOMO','BUBU']
    for name in names: print(json.dumps(compile_character(name),indent=2))

if __name__=='__main__': main()
