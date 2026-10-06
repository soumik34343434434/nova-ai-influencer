from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(os.environ.get('NOVA_REPO', Path(__file__).resolve().parents[2])).resolve()
sys.path.insert(0, str(ROOT))
from app.providers.cartoon_pose_library import DEFAULT_ACTION_POSES, build_pose_library
ASSET = ROOT / 'artifacts' / 'cartoon' / 'blender' / 'pip_cutout_layers'
PROFILE_FILE = ROOT / 'config' / 'cartoon_blender_profiles.json'
CHARACTER = 'PIP'
PROFILE = {}
OUTDIR = ROOT / 'artifacts' / 'cartoon' / 'blender' / 'pip_cutout_rig'
BLEND = ROOT / 'artifacts' / 'cartoon' / 'blender' / 'pip_cutout_rig_v2.blend'
W,H=1080,1920
PX=108.0
WORLD_H=H/PX
WORLD_W=W/PX

BASE={}
TARGETS={}
FACE_PROFILE={}
BEATS_DEFAULT=[('setup',1,42),('escalation',43,88),('reaction',89,134),('punchline',135,190)]


def px(p): return Vector(((p[0]-W/2)/PX,(H/2-p[1])/PX,0.0))


def clear():
 bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)


def image_material(name,path):
 mat=bpy.data.materials.new(name); mat.use_nodes=True
 try: mat.surface_render_method='DITHERED'
 except AttributeError: pass
 n=mat.node_tree.nodes; l=mat.node_tree.links; n.clear()
 out=n.new('ShaderNodeOutputMaterial'); tex=n.new('ShaderNodeTexImage'); tex.image=bpy.data.images.load(str(path),check_existing=True)
 sh=n.new('ShaderNodeEmission'); sh.inputs['Strength'].default_value=1.0
 tr=n.new('ShaderNodeBsdfTransparent'); mix=n.new('ShaderNodeMixShader')
 l.new(tex.outputs['Color'],sh.inputs['Color']); l.new(tex.outputs['Alpha'],mix.inputs[0]); l.new(tr.outputs[0],mix.inputs[1]); l.new(sh.outputs[0],mix.inputs[2]); l.new(mix.outputs[0],out.inputs[0])
 return mat


def color_material(name,rgb):
 mat=bpy.data.materials.new(name); mat.use_nodes=True
 n=mat.node_tree.nodes; l=mat.node_tree.links; n.clear(); out=n.new('ShaderNodeOutputMaterial'); sh=n.new('ShaderNodeEmission'); sh.inputs['Color'].default_value=(*rgb,1); sh.inputs['Strength'].default_value=1.0; l.new(sh.outputs[0],out.inputs[0]); return mat


def make_grid(name,path,segments_x=18,segments_y=24,z=0):
 meta=json.loads((ASSET/'manifest.json').read_text()); x,y,w,h=meta['parts'][name]; x1=x+w; y1=y+h
 verts=[]; uvs=[]; faces=[]
 for j in range(segments_y+1):
  v=j/segments_y; yy=y+(y1-y)*v
  for i in range(segments_x+1):
   u=i/segments_x; xx=x+(x1-x)*u; p=px((xx,yy)); verts.append((p.x,p.y,z)); uvs.append((xx/W,1.0-yy/H))
 for j in range(segments_y):
  for i in range(segments_x):
   a=j*(segments_x+1)+i; faces.append((a,a+1,a+segments_x+2,a+segments_x+1))
 mesh=bpy.data.meshes.new(name+'Mesh'); mesh.from_pydata(verts,[],faces); mesh.update(); uv=mesh.uv_layers.new(name='UVMap')
 for poly in mesh.polygons:
  for li in poly.loop_indices: uv.data[li].uv=uvs[mesh.loops[li].vertex_index]
 obj=bpy.data.objects.new('PIP_'+name,mesh); bpy.context.collection.objects.link(obj); obj.data.materials.append(image_material('MAT_'+name,path)); return obj


def create_armature():
 bpy.ops.object.armature_add(enter_editmode=True,location=(0,0,0)); arm=bpy.context.object; arm.name=f'{CHARACTER}_Rig'; eb=arm.data.edit_bones; eb.remove(eb[0]); bones={}
 def add(name,head,tail,parent=None,connect=False):
  b=eb.new(name); b.head=px(head); b.tail=px(tail); b.parent=bones.get(parent); b.use_connect=connect; bones[name]=b
 add('root',BASE['root'],BASE['neck']); add('head',BASE['neck'],BASE['head_center'],'root');
 add('arm.L',BASE['left_shoulder'],BASE['left_elbow'],'root'); add('forearm.L',BASE['left_elbow'],BASE['left_wrist'],'arm.L',True); add('hand.L',BASE['left_wrist'],(BASE['left_wrist'][0]-28,BASE['left_wrist'][1]+18),'forearm.L',True)
 add('arm.R',BASE['right_shoulder'],BASE['right_elbow'],'root'); add('forearm.R',BASE['right_elbow'],BASE['right_wrist'],'arm.R',True); add('hand.R',BASE['right_wrist'],(BASE['right_wrist'][0]+28,BASE['right_wrist'][1]+18),'forearm.R',True)
 add('backpack',(250,1135),(250,1000),'root')
 bpy.ops.object.mode_set(mode='POSE')
 for b in arm.pose.bones: b.rotation_mode='XYZ'
 bpy.ops.object.mode_set(mode='OBJECT'); return arm


def distance_weight(p,a,b):
 pa,pb=px(a),px(b); ab=pb-pa; t=max(0,min(1,(p-pa).dot(ab)/max(1e-8,ab.length_squared))); return 1-t,t


def bind(obj,arm,part):
 names=('root','head','arm.L','forearm.L','hand.L','arm.R','forearm.R','hand.R','backpack')
 groups={n:obj.vertex_groups.new(name=n) for n in names}
 for v in obj.data.vertices:
  p=Vector((v.co.x,v.co.y,0)); pp=(p.x*PX+W/2,H/2-p.y*PX)
  if part=='head': groups['head'].add([v.index],1,'REPLACE')
  elif part=='backpack': groups['backpack'].add([v.index],1,'REPLACE')
  elif part in ('left_arm','right_arm'):
   side='L' if part=='left_arm' else 'R'
   shoulder=BASE['left_shoulder'] if side=='L' else BASE['right_shoulder']
   elbow=BASE['left_elbow'] if side=='L' else BASE['right_elbow']
   wrist=BASE['left_wrist'] if side=='L' else BASE['right_wrist']
   # Smoothly partition the silhouette along the actual shoulder->elbow->wrist
   # path. This gives upper-arm, forearm and hand controls without duplicating
   # raster pixels between layers.
   pxy=pp
   def seg_t(a,b,p):
    ax,ay=a; bx,by=b; vx=bx-ax; vy=by-ay; den=max(1.0,vx*vx+vy*vy); return max(0.0,min(1.0,((p[0]-ax)*vx+(p[1]-ay)*vy)/den))
   t1=seg_t(shoulder,elbow,pxy); t2=seg_t(elbow,wrist,pxy)
   d1=((pxy[0]-elbow[0])**2+(pxy[1]-elbow[1])**2)**0.5
   span=max(80.0,((shoulder[0]-elbow[0])**2+(shoulder[1]-elbow[1])**2)**0.5)
   # Global progress is biased by nearest segment so transparent border vertices
   # do not create wild weights.
   if d1 < span*0.18:
    fore=.65; upper=.35; hand=0.0
   else:
    proj=((pxy[0]-shoulder[0])*(wrist[0]-shoulder[0])+(pxy[1]-shoulder[1])*(wrist[1]-shoulder[1]))/max(1.0,(wrist[0]-shoulder[0])**2+(wrist[1]-shoulder[1])**2)
    g=max(0.0,min(1.0,proj))
    hand=max(0.0,(g-0.72)/0.28); fore=max(0.0,min(1.0,1.0-abs(g-0.58)/0.22)); upper=max(0.0,1.0-g)
    total=max(1e-6,upper+fore+hand); upper/=total; fore/=total; hand/=total
   groups[f'arm.{side}'].add([v.index],upper,'REPLACE')
   groups[f'forearm.{side}'].add([v.index],fore,'REPLACE')
   groups[f'hand.{side}'].add([v.index],hand,'REPLACE')
  else: groups['root'].add([v.index],1,'REPLACE')
 mod=obj.modifiers.new('PIP_Armature_Deform','ARMATURE'); mod.object=arm; mod.use_deform_preserve_volume=False


def bone_parent_keep_world(obj,arm,bone):
 world=obj.matrix_world.copy(); obj.parent=arm; obj.parent_type='BONE'; obj.parent_bone=bone; obj.matrix_world=world


def make_ellipse(name,center,rx,ry,mat,z=.4,n=48):
 c=px(center); pts=[(c.x+rx*math.cos(math.tau*i/n),c.y+ry*math.sin(math.tau*i/n)) for i in range(n)]; cx=sum(x for x,y in pts)/n; cy=sum(y for x,y in pts)/n
 verts=[(x-cx,y-cy,z) for x,y in pts]; mesh=bpy.data.meshes.new(name+'Mesh'); mesh.from_pydata(verts,[],[tuple(range(n))]); mesh.update(); obj=bpy.data.objects.new(name,mesh); bpy.context.collection.objects.link(obj); obj.location=(cx,cy,0); obj.data.materials.append(mat); return obj


def make_image_patch(name,path,box,z=.4):
 x0,y0,x1,y1=box; cx=(x0+x1)/2; cy=(y0+y1)/2; hw=(x1-x0)/2/PX; hh=(y1-y0)/2/PX
 verts=[(-hw,-hh,z),(hw,-hh,z),(hw,hh,z),(-hw,hh,z)]; mesh=bpy.data.meshes.new(name+'Mesh'); mesh.from_pydata(verts,[],[(0,1,2,3)]); mesh.update(); uv=mesh.uv_layers.new(name='UVMap')
 uvs=[(x0/W,1-y0/H),(x1/W,1-y0/H),(x1/W,1-y1/H),(x0/W,1-y1/H)]
 for li,uvv in enumerate(uvs): uv.data[li].uv=uvv
 obj=bpy.data.objects.new(name,mesh); bpy.context.collection.objects.link(obj); obj.location=px((cx,cy)); obj.data.materials.append(image_material(name+'Mat',path)); return obj


def face_rig(arm):
 if FACE_PROFILE.get('mode')=='organic':
  assets=FACE_PROFILE['assets']; f=FACE_PROFILE
  eye_l=make_image_patch('Eye.L',ASSET/assets['eye_left'],f['eye_left'],.35); eye_r=make_image_patch('Eye.R',ASSET/assets['eye_right'],f['eye_right'],.35); mouth=make_image_patch('Mouth',ASSET/assets['mouth'],f['mouth'],.36)
  return {'mode':'organic','screen':None,'eyes':(eye_l,eye_r),'pupils':(None,None),'mouth':mouth}
 dark=color_material('FaceScreen',(0.025,0.045,0.065)); white=color_material('FaceLED',(0.95,1.0,1.0)); black=color_material('FacePupil',(0.01,0.015,0.02))
 f=FACE_PROFILE
 screen=make_ellipse('FaceScreen',f['screen'][:2],f['screen'][2],f['screen'][3],dark,.31)
 eye_l=make_ellipse('Eye.L',f['eye_left'][:2],f['eye_left'][2],f['eye_left'][3],white,.35); eye_r=make_ellipse('Eye.R',f['eye_right'][:2],f['eye_right'][2],f['eye_right'][3],white,.35)
 pup_l=make_ellipse('Pupil.L',f['pupil_left'][:2],f['pupil_left'][2],f['pupil_left'][3],black,.37); pup_r=make_ellipse('Pupil.R',f['pupil_right'][:2],f['pupil_right'][2],f['pupil_right'][3],black,.37)
 mouth=make_ellipse('Mouth',f['mouth'][:2],f['mouth'][2],f['mouth'][3],white,.36)
 return {'screen':screen,'eyes':(eye_l,eye_r),'pupils':(pup_l,pup_r),'mouth':mouth}


def pose_spec(a,b,t):
 t=float(t); out={}
 for k in ('root','neck','head_center','left_shoulder','left_elbow','left_wrist','right_shoulder','right_elbow','right_wrist'):
  out[k]=tuple(a[k][i]+(b[k][i]-a[k][i])*t for i in (0,1))
 for k in ('head_rot','root_rot','eye_open'):
  out[k]=a[k]+(b[k]-a[k])*t
 out['gaze']=tuple(a['gaze'][i]+(b['gaze'][i]-a['gaze'][i])*t for i in (0,1)); out['mouth']=b['mouth'] if t>=.65 else a['mouth']; return out


def write_pose(arm,frame,spec):
 bpy.context.scene.frame_set(frame)
 root=arm.pose.bones['root']; root.location=px(spec['root'])-px(BASE['root']); root.rotation_euler.z=math.radians(spec['root_rot']); root.keyframe_insert('location',frame=frame); root.keyframe_insert('rotation_euler',index=2,frame=frame)
 for side,s,e,w,ab,fb in [('L','left_shoulder','left_elbow','left_wrist','arm.L','forearm.L'),('R','right_shoulder','right_elbow','right_wrist','arm.R','forearm.R')]:
  p1,p2=arm.pose.bones[ab],arm.pose.bones[fb]
  base_a=math.atan2(px(BASE[e]).y-px(BASE[s]).y,px(BASE[e]).x-px(BASE[s]).x); base_b=math.atan2(px(BASE[w]).y-px(BASE[e]).y,px(BASE[w]).x-px(BASE[e]).x)
  ta=math.atan2(px(spec[e]).y-px(spec[s]).y,px(spec[e]).x-px(spec[s]).x); tb=math.atan2(px(spec[w]).y-px(spec[e]).y,px(spec[w]).x-px(spec[e]).x)
  p1.rotation_euler.z=ta-base_a; p2.rotation_euler.z=(tb-ta)-(base_b-base_a); p3=arm.pose.bones[f'hand.{side}']; p3.rotation_euler.z=(tb-ta)*0.12; p1.keyframe_insert('rotation_euler',index=2,frame=frame); p2.keyframe_insert('rotation_euler',index=2,frame=frame); p3.keyframe_insert('rotation_euler',index=2,frame=frame)
 head=arm.pose.bones['head']; head.rotation_euler.z=math.radians(spec['head_rot']); head.keyframe_insert('rotation_euler',index=2,frame=frame)
 scale=1.0+({'setup':0,'escalation':0.035,'reaction':-0.035,'punchline':-0.01}.get(spec.get('_beat','setup'),0)); root.scale=(1/scale,scale,1); root.keyframe_insert('scale',frame=frame)
 bp=arm.pose.bones['backpack']; bp.rotation_euler.z=math.radians(spec['root_rot']*.45); bp.keyframe_insert('rotation_euler',index=2,frame=frame)


def key_face(face,frame,spec):
 c=px(spec['head_center']); base_c=px(BASE['head_center']); a=math.radians(spec['head_rot']); co,si=math.cos(a),math.sin(a)
 if face.get('mode')=='organic':
  f=FACE_PROFILE; offsets={'Eye.L':px(((f['eye_left'][0]+f['eye_left'][2])//2,(f['eye_left'][1]+f['eye_left'][3])//2))-base_c,'Eye.R':px(((f['eye_right'][0]+f['eye_right'][2])//2,(f['eye_right'][1]+f['eye_right'][3])//2))-base_c,'Mouth':px(((f['mouth'][0]+f['mouth'][2])//2,(f['mouth'][1]+f['mouth'][3])//2))-base_c}
  def wp(v): return Vector((c.x+co*v.x-si*v.y,c.y+si*v.x+co*v.y,0))
  for obj,name in [(face['eyes'][0],'Eye.L'),(face['eyes'][1],'Eye.R'),(face['mouth'],'Mouth')]:
   obj.location=wp(offsets[name]); obj.rotation_euler.z=a; obj.keyframe_insert('location',frame=frame); obj.keyframe_insert('rotation_euler',index=2,frame=frame)
  for i,(eye,sign) in enumerate(zip(face['eyes'],(-1,1))):
   eye.scale.y=spec['eye_open']; eye.location.x += spec['gaze'][0]*0.075*sign; eye.location.y += spec['gaze'][1]*0.055; eye.keyframe_insert('scale',index=1,frame=frame); eye.keyframe_insert('location',frame=frame)
  mouth=face['mouth']; mouth.scale=(1, {'smile':1.0,'open':1.28,'frown':0.80,'flat':0.68}[spec['mouth']],1); mouth.keyframe_insert('scale',frame=frame); return
 f=FACE_PROFILE
 offsets={'screen':px(f['screen'][:2])-base_c,'Eye.L':px(f['eye_left'][:2])-base_c,'Eye.R':px(f['eye_right'][:2])-base_c,'Pupil.L':px(f['pupil_left'][:2])-base_c,'Pupil.R':px(f['pupil_right'][:2])-base_c,'Mouth':px(f['mouth'][:2])-base_c}
 def world_pos(v): return (c.x + co*v.x-si*v.y, c.y + si*v.x+co*v.y, 0)
 for name,obj in [('screen',face['screen']),('Eye.L',face['eyes'][0]),('Eye.R',face['eyes'][1]),('Pupil.L',face['pupils'][0]),('Pupil.R',face['pupils'][1]),('Mouth',face['mouth'])]:
  obj.location=world_pos(offsets[name]); obj.rotation_euler.z=a; obj.keyframe_insert('location',frame=frame); obj.keyframe_insert('rotation_euler',index=2,frame=frame)
 for eye,pupil,side in [(face['eyes'][0],face['pupils'][0],-1),(face['eyes'][1],face['pupils'][1],1)]:
  eye.scale.y=spec['eye_open']; pupil.scale.y=max(.12,spec['eye_open']); pupil.location.x=world_pos(offsets['Pupil.L' if side<0 else 'Pupil.R'])[0]+spec['gaze'][0]*.18; pupil.location.y=world_pos(offsets['Pupil.L' if side<0 else 'Pupil.R'])[1]+spec['gaze'][1]*.14; eye.keyframe_insert('scale',index=1,frame=frame); pupil.keyframe_insert('scale',index=1,frame=frame); pupil.keyframe_insert('location',frame=frame)
 mouth=face['mouth']; mouth.scale={'smile':(1,1,1),'open':(.90,1.70,1),'frown':(.88,.60,1),'flat':(.82,.30,1)}[spec['mouth']]; mouth.keyframe_insert('scale',frame=frame)


def load_voice_rows(voice_json):
    if not voice_json:
        return []
    path=Path(voice_json)
    if not path.exists():
        return []
    try:
        data=json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    return list((data.get('audio_sync',{}).get('voice_performance') or data.get('voice_performance') or []))


def build_beats(frames, voice_json=None):
    rows=load_voice_rows(voice_json)
    if not rows:
        return BEATS_DEFAULT
    durations=[float(r.get('duration',0.0) or 0.0) for r in rows[:4]]
    # Preserve enough reaction tail for each line while making frame positions track
    # the real voice performance clock.
    tails=[0.48,0.58,0.60,0.82]
    weights=[max(1.2,d+t) for d,t in zip(durations,tails)]
    total=sum(weights) or 1.0
    ranges=[]; start=1
    names=['setup','escalation','reaction','punchline']
    for i,name in enumerate(names):
        end=frames if i==len(names)-1 else start+max(18,round((frames-1)*weights[i]/total))
        ranges.append((name,start,min(frames,end)))
        start=min(frames,end+1)
    return ranges


def pose_for_index(beat, index, rows):
    row=rows[index] if index < len(rows) else {}
    action=row.get('action')
    mapped=PROFILE.get('_pose_map',{}).get(action) if action else None
    if not mapped:
        mapped=PROFILE.get('_pose_map',{}).get(beat,beat)
    return PROFILE.get('_pose_library',{}).get(mapped,TARGETS.get(beat,BASE))


def camera_performance_keys(arm, cam, beats):
    """Key a damped camera from the same character root clock."""
    for beat,start,end in beats:
        frames=(start, max(start,start+(end-start)//2), end)
        for frame in sorted(set(frames)):
            bpy.context.scene.frame_set(frame)
            root=arm.pose.bones['root']
            loc=root.matrix.to_translation()
            cam.location.x=loc.x*0.24
            cam.location.y=loc.y*0.18
            cam.location.z=30.0
            # Tiny impact punch-in; keep the stage readable.
            zoom=0.985 if beat in ('escalation','reaction') and frame>start+(end-start)//3 else 1.0
            cam.data.ortho_scale=WORLD_H*(zoom if zoom<1 else 1.0)
            cam.rotation_euler.z=root.rotation_euler.z*0.035
            cam.keyframe_insert('location',frame=frame)
            cam.keyframe_insert('rotation_euler',index=2,frame=frame)
            cam.data.keyframe_insert('ortho_scale',frame=frame)


def load_profile(character='PIP', profile_path=None):
    path=Path(profile_path) if profile_path else PROFILE_FILE
    data=json.loads(path.read_text(encoding='utf-8'))
    try:
        profile=dict(data['characters'][character])
    except KeyError as exc:
        raise ValueError(f'No Blender performance profile for character: {character}') from exc
    base=dict(profile['base'])
    targets={}
    for name in ('setup','escalation','reaction','punchline'):
        merged=dict(base); merged.update(profile.get('targets',{}).get(name,{ })); targets[name]=merged
    pose_library=build_pose_library(base,character)
    # Profile mappings override the semantic defaults when a character needs a
    # different acting beat; every named pose remains reusable by action.
    pose_map=dict({k:v for k,v in {'setup':'idle','escalation':'shock','reaction':'overwhelmed','punchline':'deadpan'}.items()})
    pose_map.update(profile.get('pose_map',{}))
    for action,pose in DEFAULT_ACTION_POSES.items():
        pose_map.setdefault(action,pose)
    profile['_pose_library']=pose_library; profile['_pose_map']=pose_map
    targets={k:pose_library.get(k,v) for k,v in targets.items()}
    return profile,base,targets,dict(profile['face'])


def profile_asset_dir(profile):
    return ROOT / profile['asset_dir']


def build(frames=190, voice_json=None, character='PIP', profile_path=None, background_path=None):
 global ASSET,BASE,TARGETS,FACE_PROFILE,CHARACTER,PROFILE
 PROFILE,BASE,TARGETS,FACE_PROFILE=load_profile(character,profile_path); CHARACTER=character; ASSET=profile_asset_dir(PROFILE)
 clear(); scene=bpy.context.scene; BEATS=build_beats(frames,voice_json); scene.render.engine='BLENDER_EEVEE'; scene.render.resolution_x=W; scene.render.resolution_y=H; scene.render.resolution_percentage=int(os.environ.get('NOVA_BLENDER_RESOLUTION','35')); scene.render.fps=24; scene.world.color=(0.0,0.0,0.0); scene.view_settings.view_transform='Standard'; scene.view_settings.look='None'; scene.view_settings.exposure=0; scene.view_settings.gamma=1
 # background
 bgmesh=bpy.data.meshes.new('BackgroundMesh'); bgmesh.from_pydata([(-WORLD_W*2,-WORLD_H*2,-.3),(WORLD_W*2,-WORLD_H*2,-.3),(WORLD_W*2,WORLD_H*2,-.3),(-WORLD_W*2,WORLD_H*2,-.3)],[],[(0,1,2,3)]); bgmesh.update(); uv=bgmesh.uv_layers.new(name='UVMap');
 for poly in bgmesh.polygons:
  for li,uvv in zip(poly.loop_indices,[(0,0),(1,0),(1,1),(0,1)]): uv.data[li].uv=uvv
 bg=bpy.data.objects.new('Background',bgmesh); bpy.context.collection.objects.link(bg); bg_path=Path(background_path) if background_path else ROOT / PROFILE['background']; bg.data.materials.append(image_material('BG',bg_path))
 arm=create_armature()
 for part,z,segx,segy in [('torso',0.0,28,34),('backpack',0.02,20,32),('left_arm',0.04,24,28),('right_arm',0.05,24,28),('head',0.10,32,34)]:
  filename=PROFILE['layers'].get(part,f'{character.lower()}_{part}.png'); obj=make_grid(part,ASSET/filename,segx,segy,z); bind(obj,arm,part)
 face=face_rig(arm)
 rows=load_voice_rows(voice_json)
 prev=pose_for_index('setup',0,rows)
 for idx,(beat,start,end) in enumerate(BEATS):
  target=pose_for_index(beat,idx,rows)
  dur=end-start
  specs=[(start,pose_spec(prev,target,-0.12)),(start+round(dur*.48),pose_spec(prev,target,1.10)),(start+round(dur*.72),target),(end,target)]
  for frame,spec in specs:
   spec=dict(spec); spec['_beat']=beat; write_pose(arm,frame,spec); key_face(face,frame,spec)
  prev=target
 # Add speech-reactive mouth shape cues: fast micro-open/close around each phrase center.
 rows=load_voice_rows(voice_json)
 for idx,(beat,start,end) in enumerate(BEATS):
  spec=pose_for_index(beat,idx,rows)
  row=rows[idx] if idx < len(rows) else {}
  activity=row.get('activity') or []
  if activity:
   for f in range(start,end+1,2):
    q=(f-start)/max(1,end-start); ai=min(len(activity)-1,int(round(q*(len(activity)-1))))
    try: energy=max(0.0,min(1.0,float(activity[ai])))
    except (TypeError,ValueError): energy=0.0
    base_m={'smile':1.0,'open':1.5,'frown':.7,'flat':.3}[spec['mouth']]
    face['mouth'].scale.y=base_m*(0.30+0.90*energy); face['mouth'].keyframe_insert('scale',frame=f)
  else:
   mid=start+round((end-start)*.45)
   for f,sy in [(mid-2,.55),(mid,.95),(mid+2,.38),(mid+4,1.0)]:
    face['mouth'].scale.y={'smile':1,'open':1.5,'frown':.7,'flat':.3}[spec['mouth']]*sy; face['mouth'].keyframe_insert('scale',frame=f)
 # Blink on non-impact moments.
 for beat,start,end in BEATS:
  f=start+round((end-start)*.34)
  for eye,pupil in zip(face['eyes'],face['pupils']):
   eye.scale.y=.10; eye.keyframe_insert('scale',index=1,frame=f); eye.scale.y=1; eye.keyframe_insert('scale',index=1,frame=f+2)
   if pupil is not None:
    pupil.scale.y=.06; pupil.keyframe_insert('scale',index=1,frame=f); pupil.scale.y=1; pupil.keyframe_insert('scale',index=1,frame=f+2)
 bpy.ops.object.camera_add(location=(0,0,30)); cam=bpy.context.object; cam.name=f'{CHARACTER}_Camera'; cam.data.type='ORTHO'; cam.data.ortho_scale=WORLD_H; scene.camera=cam
 camera_performance_keys(arm,cam,BEATS)
 # Blender's keyed transforms use its default BEZIER interpolation, providing
 # real continuous in-betweens between performance poses.
 scene.frame_start=1; scene.frame_end=frames; scene.render.image_settings.file_format='PNG'; scene.render.filepath=str(OUTDIR/'frame_####')
 effective_rows=load_voice_rows(voice_json); effective_shot_durations=[round(max(1.0,float(r.get('duration',0.0) or 0.0))+tail,3) for r,tail in zip(effective_rows[:4],(0.48,0.58,0.60,0.82))] if rows else [round((b-a)/24,3) for _,a,b in BEATS]
 meta={'engine':'blender-cutout-performance-rig-v3','character':CHARACTER,'beats':BEATS,'voice_json':str(voice_json) if voice_json else None,'effective_shot_durations':effective_shot_durations,'actions':[r.get('action') for r in effective_rows[:4]],'resolution':[W,H],'fps':24,'frames':frames,'deformable_layers':['torso','head','left_arm','right_arm','backpack'],'armature_bones':['root','head','arm.L','forearm.L','hand.L','arm.R','forearm.R','hand.R','backpack'],'facial_controls':['blink','gaze','mouth_shape','speech_micro_motion','organic_face_patches'],'pose_library':list(PROFILE.get('_pose_library',{}).keys()),'pose_map':dict(PROFILE.get('_pose_map',{})),'in_between':'overshoot-and-settle-bezier','source':'clean-pip-master-cutout'}
 t=bpy.data.texts.new('NOVA_BLENDER_RIG_METADATA.json'); t.write(json.dumps(meta,indent=2)); return scene,meta


def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--build',action='store_true'); ap.add_argument('--render',action='store_true'); ap.add_argument('--frames',type=int,default=190); ap.add_argument('--blend-output',type=Path,default=BLEND); ap.add_argument('--voice-json',type=Path,default=None); ap.add_argument('--character',default='PIP'); ap.add_argument('--profile',type=Path,default=None); ap.add_argument('--background',type=Path,default=None); args=ap.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
 OUTDIR.mkdir(parents=True,exist_ok=True); scene,meta=build(args.frames,args.voice_json,args.character,args.profile,args.background); bpy.ops.wm.save_as_mainfile(filepath=str(args.blend_output))
 if args.render:
  selected=os.environ.get('NOVA_KEY_FRAMES','').strip()
  if selected:
   for token in selected.split(','):
    f=int(token.strip()); scene.frame_set(f); scene.render.filepath=str(OUTDIR/f'key_{f:04d}.png'); bpy.ops.render.render(write_still=True)
  else:
   bpy.ops.render.render(animation=True)
if __name__=='__main__': main()
