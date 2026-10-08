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
from app.providers.cartoon_pose_library import DEFAULT_ACTION_POSES, build_pose_library, make_action_pose_chain
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
 if FACE_PROFILE.get('mode')=='static':
  # Keep a hidden compatibility face rig so the downstream speech/blink
  # performance code can still run without ever compositing patches over MOMO's
  # intact organic face.
  saved_mode=FACE_PROFILE.get('mode')
  FACE_PROFILE['mode']='organic'
  compat=face_rig(arm)
  FACE_PROFILE['mode']=saved_mode
  def hide(node):
   if isinstance(node, dict):
    for child in node.values(): hide(child)
   elif isinstance(node, (tuple,list)):
    for child in node: hide(child)
   elif node is not None:
    try: node.hide_render=True
    except AttributeError: pass
  hide(compat)
  compat['mode']='static'
  return compat
 if FACE_PROFILE.get('mode')=='organic':
  assets=FACE_PROFILE['assets']; f=FACE_PROFILE
  eye_l=make_image_patch('Eye.L',ASSET/assets['eye_left'],f['eye_left'],.35); eye_r=make_image_patch('Eye.R',ASSET/assets['eye_right'],f['eye_right'],.35); mouth=make_image_patch('Mouth',ASSET/assets['mouth'],f['mouth'],.36)
  brow_mat=color_material('OrganicBrow',(0.16,0.09,0.05) if CHARACTER=='MOMO' else (0.05,0.16,0.22))
  brows=[]
  for name,box in (('Brow.L',f['eye_left']),('Brow.R',f['eye_right'])):
   x0,y0,x1,y1=box; cx=(x0+x1)/2; cy=y0-24
   brow=prop_box(name,((cx-W/2)/PX,(H/2-cy)/PX),(0.46,0.075,0.035),brow_mat,z=.37); brows.append(brow)
  return {'mode':'organic','screen':None,'eyes':(eye_l,eye_r),'pupils':(None,None),'mouth':mouth,'brows':tuple(brows)}
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


def action_pose_chain(prev, target, beat, action, start, end):
    """Adapt the shared physical action-chain primitive to absolute frame numbers."""
    profiles=PROFILE.get('transition_profiles') or {}
    chain=make_action_pose_chain(prev,target,action,span=max(8,end-start),transition_profiles=profiles)
    root_dx=float(target['root'][0]-prev['root'][0]); root_dy=float(target['root'][1]-prev['root'][1])
    wrist_dx=sum(float(target[k][0]-prev[k][0]) for k in ('left_wrist','right_wrist'))/2.0
    wrist_dy=sum(float(target[k][1]-prev[k][1]) for k in ('left_wrist','right_wrist'))/2.0
    dx=wrist_dx*0.72+root_dx*0.28; dy=wrist_dy*0.72+root_dy*0.28
    return [(start+offset,{**spec,'_motion':(dx,dy),'_action':action},phase) for offset,spec,phase in chain]


def write_pose(arm,frame,spec):
 bpy.context.scene.frame_set(frame)
 transition=PROFILE.get('transition_profiles') or {}
 default_transition=transition.get('default',{})
 action_transition=transition.get(spec.get('_action',''),{})
 secondary=dict(default_transition.get('secondary',{}))
 secondary.update(action_transition.get('secondary',{}))
 joint_lag=max(0,int(secondary.get('joint_lag_frames',1)))
 head_lag=max(0,int(secondary.get('head_lag_frames',2)))
 hand_lag=max(0,int(secondary.get('hand_lag_frames',3)))
 accessory_lag=max(0,int(secondary.get('accessory_lag_frames',4)))
 max_frame=int(bpy.context.scene.frame_end)
 def kframe(offset):
  return min(max_frame,max(1,int(frame)+int(offset)))
 motion=spec.get('_motion',(0.0,0.0))
 motion_len=max(1.0,math.hypot(float(motion[0]),float(motion[1])))
 ux,uy=float(motion[0])/motion_len,float(motion[1])/motion_len
 drive=max(-1.0,min(1.0,float(motion[0])/140.0 + float(motion[1])/220.0*0.45))
 phase=spec.get('_phase','settle')
 phase_weight={'approach':0.0,'anticipation':0.32,'contact_approach':0.68,'contact':1.0,'overshoot':1.25,'recoil':0.82,'settle':0.20}.get(phase,0.0)
 impulse_phase={'approach':0.0,'anticipation':-0.26,'contact_approach':0.48,'contact':1.0,'overshoot':1.34,'recoil':-0.78,'settle':0.08}.get(phase,0.0)
 impact_translate=float(action_transition.get('impact_translate_px',default_transition.get('impact_translate_px',18.0)))
 impulse=impact_translate*impulse_phase
 root=arm.pose.bones['root']
 root.location=px(spec['root'])-px(BASE['root'])
 root.location.x += ux*(impulse/PX)
 root.location.y += uy*(impulse/PX)
 root.rotation_euler.z=math.radians(spec['root_rot'] + drive*impulse_phase*3.2)
 root.keyframe_insert('location',frame=frame)
 root.keyframe_insert('rotation_euler',index=2,frame=frame)
 for side,s,e,w,ab,fb in [('L','left_shoulder','left_elbow','left_wrist','arm.L','forearm.L'),('R','right_shoulder','right_elbow','right_wrist','arm.R','forearm.R')]:
  p1,p2=arm.pose.bones[ab],arm.pose.bones[fb]
  base_a=math.atan2(px(BASE[e]).y-px(BASE[s]).y,px(BASE[e]).x-px(BASE[s]).x)
  base_b=math.atan2(px(BASE[w]).y-px(BASE[e]).y,px(BASE[w]).x-px(BASE[e]).x)
  ta=math.atan2(px(spec[e]).y-px(spec[s]).y,px(spec[e]).x-px(spec[s]).x)
  tb=math.atan2(px(spec[w]).y-px(spec[e]).y,px(spec[w]).x-px(spec[e]).x)
  p1.rotation_euler.z=ta-base_a
  p2.rotation_euler.z=(tb-ta)-(base_b-base_a)
  p3=arm.pose.bones[f'hand.{side}']
  p3.rotation_euler.z=(tb-ta)*0.12
  arm_frame=kframe(joint_lag)
  p1.keyframe_insert('rotation_euler',index=2,frame=arm_frame)
  p2.keyframe_insert('rotation_euler',index=2,frame=arm_frame)
  p3.keyframe_insert('rotation_euler',index=2,frame=kframe(hand_lag))
 head_lag_deg=float(secondary.get('head_lag_deg',0.0))
 hand_follow_deg=float(secondary.get('hand_follow_deg',0.0))
 accessory_follow_deg=float(secondary.get('accessory_follow_deg',0.0))
 head=arm.pose.bones['head']
 head.rotation_euler.z=math.radians(spec['head_rot']-head_lag_deg*drive*phase_weight)
 head.location.x=-ux*min(0.028,abs(impulse)/9000.0)
 head.location.y=-uy*min(0.028,abs(impulse)/9000.0)
 head.keyframe_insert('rotation_euler',index=2,frame=kframe(head_lag))
 head.keyframe_insert('location',frame=kframe(head_lag))
 hand_follow=math.radians(hand_follow_deg*drive*phase_weight)
 for side in ('L','R'):
  hand=arm.pose.bones[f'hand.{side}']
  hand.rotation_euler.z += hand_follow*(1.0 if side=='R' else -1.0)
  hand.keyframe_insert('rotation_euler',index=2,frame=kframe(hand_lag))
 beat_scale={'setup':1.0,'escalation':1.035,'reaction':0.965,'punchline':0.99}.get(spec.get('_beat','setup'),1.0)
 phase_scale={'approach':1.0,'anticipation':0.975,'contact_approach':1.015,'contact':1.055,'overshoot':0.94,'recoil':1.02,'settle':1.0}.get(phase,1.0)
 scale=beat_scale*phase_scale
 root.scale=(1/scale,scale,1)
 root.keyframe_insert('scale',frame=frame)
 bp=arm.pose.bones['backpack']
 bp.rotation_euler.z=math.radians(spec['root_rot']*.45-accessory_follow_deg*drive*phase_weight)
 bp.location.x=-ux*min(0.055,abs(impulse)/4200.0)
 bp.location.y=-uy*min(0.055,abs(impulse)/4200.0)
 bp.keyframe_insert('rotation_euler',frame=kframe(accessory_lag))
 bp.keyframe_insert('location',frame=kframe(accessory_lag))

def key_face(face,frame,spec):
 if face.get('mode')=='static':
  return
 phase=spec.get('_phase','settle')
 c=px(spec['head_center']); base_c=px(BASE['head_center']); a=math.radians(spec['head_rot']); co,si=math.cos(a),math.sin(a)
 if face.get('mode')=='organic':
  f=FACE_PROFILE; action=spec.get('_action','')
  contact_side={'press_button':1,'give_homework':-1,'tap_alarm':1,'open_fridge':1,'tap_shoes':1,'shock':-1,'overwhelmed':1}.get(action,0)
  offsets={'Eye.L':px(((f['eye_left'][0]+f['eye_left'][2])//2,(f['eye_left'][1]+f['eye_left'][3])//2))-base_c,'Eye.R':px(((f['eye_right'][0]+f['eye_right'][2])//2,(f['eye_right'][1]+f['eye_right'][3])//2))-base_c,'Mouth':px(((f['mouth'][0]+f['mouth'][2])//2,(f['mouth'][1]+f['mouth'][3])//2))-base_c}
  def wp(v): return Vector((c.x+co*v.x-si*v.y,c.y+si*v.x+co*v.y,0))
  for obj,name in [(face['eyes'][0],'Eye.L'),(face['eyes'][1],'Eye.R'),(face['mouth'],'Mouth')]:
   obj.location=wp(offsets[name]); obj.rotation_euler.z=a; obj.keyframe_insert('location',frame=frame+1); obj.keyframe_insert('rotation_euler',index=2,frame=frame+1)
  impact_eye=float((PROFILE.get('transition_profiles') or {}).get('default',{}).get('secondary',{}).get('impact_eye_open',0.06))
  reaction_boost=impact_eye*(1.0 if phase=='contact' else 0.65 if phase=='overshoot' else 0.35 if phase=='recoil' else 0.0)
  for eye,sign in zip(face['eyes'],(-1,1)):
   # The eye on the acting-hand side leads the impact reaction; the far eye
   # follows more softly, keeping the face from reading as a uniform blink.
   side_weight=1.25 if contact_side and sign==contact_side else 0.75 if contact_side else 1.0
   eye_open=max(0.35,min(1.35,spec['eye_open']+reaction_boost*side_weight))
   asym=0.07*sign if action in ('deadpan','overwhelmed') else 0.045*sign
   if action in ('shock','tap_alarm'): eye_open=min(1.35,eye_open+0.08)
   eye.scale.x=1.0+asym; eye.scale.y=eye_open
   eye.location.x += spec['gaze'][0]*0.11*sign; eye.location.y += spec['gaze'][1]*0.085
   eye.keyframe_insert('scale',frame=frame); eye.keyframe_insert('location',frame=frame)
  if 'brows' in face:
   brow_l,brow_r=face['brows']
   brow_l.rotation_euler.z=a+math.radians(0.08*spec['head_rot'] + (8.0 if action=='overwhelmed' else 3.0 if action=='deadpan' else 0.0))
   brow_r.rotation_euler.z=a+math.radians(0.08*spec['head_rot'] + (-8.0 if action=='overwhelmed' else -3.0 if action=='deadpan' else 0.0))
   lift=0.14 if action=='shock' or phase in ('contact','overshoot') else -0.06 if action in ('deadpan','overwhelmed') else 0.0
   brow_l.location=wp(offsets['Eye.L']); brow_r.location=wp(offsets['Eye.R'])
   brow_l.location.y += lift; brow_r.location.y += lift
   brow_l.keyframe_insert('location',frame=frame+1); brow_r.keyframe_insert('location',frame=frame+1)
   brow_l.keyframe_insert('rotation_euler',index=2,frame=frame+1); brow_r.keyframe_insert('rotation_euler',index=2,frame=frame+1)
  mouth=face['mouth']
  mouth.scale={'smile':(1.06,1.0,1),'open':(.90,1.45,1),'frown':(.94,.68,1),'flat':(.84,.44,1)}[spec['mouth']]
  mouth.rotation_euler.z=a+math.radians(-2.5 if spec['mouth']=='smile' else 3.0 if spec['mouth']=='frown' else 0.0)
  mouth.keyframe_insert('scale',frame=frame); mouth.keyframe_insert('rotation_euler',index=2,frame=frame)
  return
 f=FACE_PROFILE
 offsets={'screen':px(f['screen'][:2])-base_c,'Eye.L':px(f['eye_left'][:2])-base_c,'Eye.R':px(f['eye_right'][:2])-base_c,'Pupil.L':px(f['pupil_left'][:2])-base_c,'Pupil.R':px(f['pupil_right'][:2])-base_c,'Mouth':px(f['mouth'][:2])-base_c}
 def world_pos(v): return (c.x + co*v.x-si*v.y, c.y + si*v.x+co*v.y, 0)
 for name,obj in [('screen',face['screen']),('Eye.L',face['eyes'][0]),('Eye.R',face['eyes'][1]),('Pupil.L',face['pupils'][0]),('Pupil.R',face['pupils'][1]),('Mouth',face['mouth'])]:
  obj.location=world_pos(offsets[name]); obj.rotation_euler.z=a; obj.keyframe_insert('location',frame=frame); obj.keyframe_insert('rotation_euler',index=2,frame=frame)
 contact_side={'press_button':1,'give_homework':-1,'tap_alarm':1,'open_fridge':1,'tap_shoes':1,'shock':-1,'overwhelmed':1}.get(spec.get('_action',''),0)
 for eye,pupil,side in [(face['eyes'][0],face['pupils'][0],-1),(face['eyes'][1],face['pupils'][1],1)]:
  impact_eye=float((PROFILE.get('transition_profiles') or {}).get('default',{}).get('secondary',{}).get('impact_eye_open',0.06)); reaction_boost=impact_eye*(1.0 if phase=='contact' else 0.65 if phase=='overshoot' else 0.35 if phase=='recoil' else 0.0); side_weight=1.25 if contact_side and side==contact_side else 0.75 if contact_side else 1.0; eye_open=max(.35,min(1.35,spec['eye_open']+reaction_boost*side_weight)); eye.scale.y=eye_open; pupil.scale.y=max(.12,eye_open); pupil.location.x=world_pos(offsets['Pupil.L' if side<0 else 'Pupil.R'])[0]+spec['gaze'][0]*.18; pupil.location.y=world_pos(offsets['Pupil.L' if side<0 else 'Pupil.R'])[1]+spec['gaze'][1]*.14; eye.keyframe_insert('scale',index=1,frame=frame); pupil.keyframe_insert('scale',index=1,frame=frame); pupil.keyframe_insert('location',frame=frame)
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


def _action_focus(action):
    return {
        'give_homework': (2.55, -1.35),
        'press_button': (2.55, -1.65),
        'shock': (1.6, 0.0),
        'overwhelmed': (-0.15, -0.08),
        'deadpan': (0.0, 0.0),
        'tap_alarm': (2.40, -1.45),
        'open_fridge': (2.75, -1.20),
        'tap_shoes': (2.65, -1.70),
    }.get(action, (0.8, -0.05))


def _performed_action_focus(action, arm, fallback):
    """Blend the authored action focus toward the currently posed contact hand."""
    hand_side = {
        'give_homework': 'L', 'press_button': 'R', 'tap_alarm': 'R',
        'open_fridge': 'R', 'tap_shoes': 'R',
    }.get(action)
    if hand_side is None:
        return fallback
    hand = arm.pose.bones.get(f'hand.{hand_side}')
    if hand is None:
        return fallback
    # The hand bone origin is at the wrist. Focus on its posed endpoint so
    # button presses, taps, and handoffs frame the actual contact point.
    point = arm.matrix_world @ hand.tail
    return (float(point.x), float(point.y))


def camera_performance_keys(arm, cam, beats, rows=None):
    """Beat-specific camera acting with smooth handoffs and readable shot-size contrast."""
    rows=rows or []
    phase_points=(0.00,0.18,0.42,0.58,0.72,0.84,1.00)
    phase_names=('approach','anticipation','contact_approach','contact','overshoot','recoil','settle')
    base_scale={'setup':11.8,'escalation':10.6,'reaction':11.2,'punchline':11.4}
    action_scale={'give_homework':0.96,'press_button':0.96,'shock':0.90,'overwhelmed':0.97,'deadpan':0.95,'tap_alarm':0.92,'open_fridge':0.94,'tap_shoes':0.92}
    previous_state=None
    for idx,(beat,start,end) in enumerate(beats):
        action=(rows[idx].get('action') if idx < len(rows) else '') or beat
        fx,fy=_action_focus(action); span=max(1,end-start)
        for frac,phase in zip(phase_points,phase_names):
            frame=min(end,max(start,round(start+span*frac))); bpy.context.scene.frame_set(frame)
            root=arm.pose.bones['root']; loc=root.matrix.to_translation()
            if previous_state is not None and phase=='approach':
                cam.location.x,cam.location.y,cam.location.z=previous_state[0]
                cam.rotation_euler.z=previous_state[1]
                cam.data.ortho_scale=previous_state[2]
                cam.keyframe_insert('location',frame=frame); cam.keyframe_insert('rotation_euler',index=2,frame=frame); cam.data.keyframe_insert('ortho_scale',frame=frame)
                continue
            focus_w={'approach':0.08,'anticipation':0.16,'contact_approach':0.38,'contact':0.62,'overshoot':0.74,'recoil':0.56,'settle':0.34}[phase]
            contact_focus=_performed_action_focus(action,arm,(fx,fy))
            # Let the performed contact point dominate at impact, then retain
            # enough of it through overshoot to make the action readable.
            focus_mix={'approach':0.0,'anticipation':0.0,'contact_approach':0.45,'contact':0.90,'overshoot':0.70,'recoil':0.25,'settle':0.0}[phase]
            focus_x=fx+(contact_focus[0]-fx)*focus_mix
            focus_y=fy+(contact_focus[1]-fy)*focus_mix
            punch={
                'setup':{'approach':0.0,'anticipation':-0.008,'contact_approach':-0.015,'contact':-0.022,'overshoot':-0.018,'recoil':0.0,'settle':0.0},
                'escalation':{'approach':0.0,'anticipation':0.0,'contact_approach':-0.014,'contact':-0.065,'overshoot':-0.085,'recoil':-0.035,'settle':-0.010},
                'reaction':{'approach':0.0,'anticipation':0.010,'contact_approach':0.020,'contact':0.008,'overshoot':0.0,'recoil':0.018,'settle':0.045},
                'punchline':{'approach':0.0,'anticipation':-0.008,'contact_approach':-0.018,'contact':-0.040,'overshoot':-0.052,'recoil':-0.020,'settle':-0.060},
            }[beat][phase]
            lateral={'setup':0.0,'escalation':0.055,'reaction':-0.075,'punchline':0.085}.get(beat,0.0)
            curve=math.sin(math.pi*frac)
            # Lock the camera kick to the performed impact instead of an
            # unrelated frame-frequency oscillation. The authored phase
            # envelope gives a clear hit, then a smaller follow-through and
            # recoil response; it cannot accidentally cancel at contact.
            shake_profile={
                'setup':{'approach':0.0,'anticipation':0.0,'contact_approach':0.0,'contact':0.0,'overshoot':0.0,'recoil':0.0,'settle':0.0},
                'escalation':{'approach':0.0,'anticipation':0.0,'contact_approach':0.0,'contact':0.010,'overshoot':0.0065,'recoil':0.0025,'settle':0.0},
                'reaction':{'approach':0.0,'anticipation':0.0,'contact_approach':0.001,'contact':0.003,'overshoot':0.002,'recoil':0.001,'settle':0.0},
                'punchline':{'approach':0.0,'anticipation':0.0,'contact_approach':0.0,'contact':0.003,'overshoot':0.002,'recoil':0.001,'settle':0.0},
            }
            shake=shake_profile[beat][phase]
            cam.location.x=loc.x*0.08 + focus_x*focus_w + lateral*curve + shake
            cam.location.y=loc.y*0.06 + focus_y*focus_w + shake*0.40
            cam.location.z=30.0
            cam.data.ortho_scale=max(10.0,base_scale[beat]*action_scale.get(action,1.0)*(1.0+punch))
            cam.rotation_euler.z=root.rotation_euler.z*0.05 + math.radians(shake*1.2)
            cam.keyframe_insert('location',frame=frame); cam.keyframe_insert('rotation_euler',index=2,frame=frame); cam.data.keyframe_insert('ortho_scale',frame=frame)
        previous_state=(tuple(cam.location),cam.rotation_euler.z,cam.data.ortho_scale)


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



def prop_materials():
    return {
        'paper': color_material('PropPaper',(0.92,0.94,0.96)),
        'device': color_material('PropDevice',(0.10,0.17,0.22)),
        'accent': color_material('PropAccent',(0.12,0.46,0.72)),
        'dark': color_material('PropDark',(0.035,0.055,0.07)),
        'warm': color_material('PropWarm',(0.76,0.58,0.22)),
        'shadow': color_material('PropShadow',(0.18,0.23,0.28)),
    }


def prop_box(name,loc,size,mat,z=0.18):
    bpy.ops.mesh.primitive_cube_add(location=(loc[0],loc[1],z))
    obj=bpy.context.object; obj.name=name; obj.scale=(size[0]/2,size[1]/2,size[2]/2); obj.data.materials.append(mat)
    return obj


def key_visibility(obj,start,end,scale=1.0):
    obj.scale=tuple(v*0.001 for v in obj.scale); obj.keyframe_insert('scale',frame=max(1,start-1))
    obj.scale=tuple(v*scale/0.001 for v in obj.scale); obj.keyframe_insert('scale',frame=start)
    obj.keyframe_insert('scale',frame=end)
    obj.scale=tuple(v*0.001 for v in obj.scale); obj.keyframe_insert('scale',frame=min(int(bpy.context.scene.frame_end),end+1))


def build_story_world(rows,beats):
    mats=prop_materials()
    # A small stage plane + floor shadow makes the character read as grounded even
    # when the source background is a soft generated plate.
    stage=prop_box('NOVA_DeskStage',(0,-6.10),(12.8,1.15,0.12),mats['device'],z=-0.04)
    trim=prop_box('NOVA_DeskTrim',(0,-5.52),(12.8,0.06,0.10),mats['accent'],z=-0.02)
    shadow=make_ellipse('NOVA_GroundShadow',(540,1478),2.25,0.33,mats['shadow'],z=-0.01,n=64)
    for obj in (stage,trim):
        obj.keyframe_insert('scale',frame=1); obj.keyframe_insert('scale',frame=int(bpy.context.scene.frame_end))
    shadow.scale=(1.0,0.90,1.0); shadow.keyframe_insert('scale',frame=1); shadow.keyframe_insert('scale',frame=int(bpy.context.scene.frame_end))
    # Persistent environment dressing: recognizable shelving and varied book spines.
    for n,sy in enumerate((-0.85,-1.75,-2.65)):
        shelf=prop_box(f'LibraryShelfBoard.{n}',(-3.75,sy),(2.55,0.12,0.22),mats['warm'],z=0.02); shelf.keyframe_insert('location',frame=1); shelf.keyframe_insert('location',frame=int(bpy.context.scene.frame_end))
    for n,sx in enumerate((-5.02,-2.48)):
        post=prop_box(f'LibraryShelfPost.{n}',(sx,-1.72),(0.14,3.55,0.22),mats['warm'],z=0.02); post.keyframe_insert('location',frame=1); post.keyframe_insert('location',frame=int(bpy.context.scene.frame_end))
    book_layout=[(-4.72,-0.55,0.28,0.72),(-4.36,-0.56,0.22,0.78),(-3.98,-0.55,0.30,0.70),(-3.52,-0.56,0.24,0.82),(-3.15,-0.55,0.26,0.74),(-4.62,-1.45,0.24,0.75),(-4.28,-1.44,0.30,0.82),(-3.90,-1.43,0.21,0.70),(-3.50,-1.44,0.27,0.78),(-3.12,-1.43,0.23,0.72),(-4.68,-2.35,0.31,0.80),(-4.25,-2.34,0.22,0.70),(-3.90,-2.35,0.28,0.76),(-3.50,-2.35,0.25,0.82),(-3.12,-2.34,0.27,0.68)]
    for n,(bx,by,bw,bh) in enumerate(book_layout):
        book=prop_box(f'LibraryBook.{n}',(bx,by),(bw,bh,0.14),mats['accent'] if n%3 else mats['device'],z=0.20); book.keyframe_insert('location',frame=1); book.keyframe_insert('location',frame=int(bpy.context.scene.frame_end))
    for name,loc,size,z,mat in (
        ('DeskLampBase',(3.55,-2.90),(0.62,0.20,0.16),0.20,mats['device']),
        ('DeskLampArm',(3.68,-2.35),(0.10,1.10,0.10),0.24,mats['accent']),
        ('DeskLampHead',(3.48,-1.82),(0.62,0.18,0.14),0.27,mats['warm']),
    ):
        lamp=prop_box(name,loc,size,mat,z=z); lamp.keyframe_insert('location',frame=1); lamp.keyframe_insert('location',frame=int(bpy.context.scene.frame_end))
    sign_font=bpy.data.curves.new('LibrarySignCurve','FONT'); sign_font.body='MOMO WORK DESK'; sign_font.align_x='CENTER'; sign_font.size=0.72; sign_font.extrude=0.008
    sign=bpy.data.objects.new('LibrarySign',sign_font); bpy.context.collection.objects.link(sign); sign.location=(0.0,2.75,-0.20); sign.data.materials.append(mats['dark']); sign.keyframe_insert('location',frame=1); sign.keyframe_insert('location',frame=int(bpy.context.scene.frame_end))
    board=prop_box('HomeworkNoticeBoard',(4.35,0.05),(1.30,1.75,0.08),mats['paper'],z=-0.18); board.keyframe_insert('location',frame=1); board.keyframe_insert('location',frame=int(bpy.context.scene.frame_end))
    for n,(dx,dy,rot) in enumerate(((-0.15,0.48,-6),(0.12,0.05,5),(-0.10,-0.42,2))):
        note=prop_box(f'HomeworkNote.{n}',(4.35+dx,0.05+dy),(0.32,0.32,0.035),mats['accent'],z=-0.02); note.rotation_euler.z=math.radians(rot); note.keyframe_insert('location',frame=1); note.keyframe_insert('rotation_euler',index=2,frame=1); note.keyframe_insert('location',frame=int(bpy.context.scene.frame_end)); note.keyframe_insert('rotation_euler',index=2,frame=int(bpy.context.scene.frame_end))
    action_rows=rows[:4] if rows else []
    action_specs={
        'give_homework': ('printer',2.20,-2.55), 'press_button': ('console',2.20,-2.65),
        'checks_phone': ('phone',2.10,-3.10), 'finish_work': ('papers',-1.20,-3.20),
        'shock': ('printer',2.05,-2.35), 'overwhelmed': ('papers',-1.75,-3.00),
        'deadpan': ('phone',2.10,-3.10), 'tap_alarm': ('alarm',2.15,-2.58),
        'open_fridge': ('fridge',2.35,-2.55), 'tap_shoes': ('shoes',2.15,-2.95),
    }
    for idx,(beat,start,end) in enumerate(beats):
        action=(action_rows[idx].get('action') if idx < len(action_rows) else '') or beat
        kind,x,y=action_specs.get(action,('book',3.0,-4.9))
        objs=[]
        if kind=='printer':
            body=prop_box(f'PropPrinter.{idx}',(x,y),(2.0,1.05,0.42),mats['device']); objs.append(body)
            slot=prop_box(f'PropPrinterSlot.{idx}',(x,y+0.42),(1.20,0.12,0.06),mats['dark']); objs.append(slot)
            tray=prop_box(f'PropPrinterTray.{idx}',(x,y-0.58),(1.55,0.28,0.12),mats['warm'],z=0.24); objs.append(tray)
            status=make_ellipse(f'PropPrinterStatus.{idx}',(W/2+x*PX,H/2-y*PX-18),0.08,0.08,mats['accent'],z=0.37,n=24); objs.append(status)
            label_font=bpy.data.curves.new(f'PrinterReadoutCurve.{idx}','FONT'); label_font.body='PRINTING 300 PAGES'; label_font.align_x='CENTER'; label_font.size=0.18; label_font.extrude=0.004
            label=bpy.data.objects.new(f'PrinterReadout.{idx}',label_font); bpy.context.collection.objects.link(label); label.location=(x,y+0.14,0.38); label.data.materials.append(mats['paper']); objs.append(label)
            paper=prop_box(f'PropPaper.{idx}',(x,y+0.73),(0.86,0.70,0.035),mats['paper'],z=0.30); objs.append(paper)
            paper.keyframe_insert('location',frame=start); paper.location.y += 0.25; paper.keyframe_insert('location',frame=round(start+(end-start)*0.32)); paper.location.y += 0.18; paper.keyframe_insert('location',frame=round(start+(end-start)*0.58)); paper.location.y += 0.12; paper.keyframe_insert('location',frame=end)
            if action=='shock':
                for n,(ox,oy,dx,dy,rz) in enumerate(((-0.72,0.15,-0.18,0.82,-18),(0.18,0.22,0.24,0.96,12),(0.86,0.04,0.36,0.72,24))):
                    burst=prop_box(f'PropFlyingPaper.{idx}.{n}',(x+ox,y+oy),(0.54,0.72,0.03),mats['paper'],z=0.29+0.01*n); burst.rotation_euler.z=math.radians(rz); objs.append(burst)
                    burst.keyframe_insert('location',frame=round(start+(end-start)*0.20)); burst.location.x += dx; burst.location.y += dy; burst.keyframe_insert('location',frame=round(start+(end-start)*0.72)); burst.location.x += dx*0.35; burst.location.y += dy*0.25; burst.keyframe_insert('location',frame=end)
        elif kind=='phone':
            body=prop_box(f'PropPhone.{idx}',(x,y),(1.35,0.68,0.10),mats['dark'],z=0.31); objs.append(body)
            screen=prop_box(f'PropPhoneScreen.{idx}',(x,y+0.01),(1.08,0.52,0.035),mats['accent'],z=0.38); objs.append(screen)
            text_font=bpy.data.curves.new(f'PhoneTextCurve.{idx}','FONT'); text_font.body='47 NEW TASKS'; text_font.align_x='CENTER'; text_font.size=0.12; text_font.extrude=0.003
            label=bpy.data.objects.new(f'PhoneText.{idx}',text_font); bpy.context.collection.objects.link(label); label.location=(x,y-0.42,0.40); label.data.materials.append(mats['paper']); objs.append(label)
            for obj in (body,screen,label):
                obj.keyframe_insert('location',frame=start); obj.keyframe_insert('location',frame=end)
        elif kind=='console':
            base=prop_box(f'PropConsole.{idx}',(x,y),(1.65,0.85,0.35),mats['device']); objs.append(base)
            button=prop_box(f'PropButton.{idx}',(x,y+0.48),(0.40,0.22,0.12),mats['accent'],z=0.36); objs.append(button)
            button.location.y += 0.06; button.keyframe_insert('location',frame=start); button.location.y -= 0.10; button.keyframe_insert('location',frame=round(start+(end-start)*0.60)); button.location.y += 0.04; button.keyframe_insert('location',frame=end)
        elif kind=='papers':
            for n,(dx,dy,rot) in enumerate(((-0.90,0.0,-12),(0.0,0.16,4),(0.92,-0.02,11))):
                sheet=prop_box(f'PropPaperScatter.{idx}.{n}',(x+dx,y+dy),(0.66,0.88,0.025),mats['paper'],z=0.20); sheet.rotation_euler.z=math.radians(rot); objs.append(sheet)
        elif kind=='book':
            for n,(dy,w) in enumerate(((0.0,1.25),(0.16,1.10),(0.31,0.96))):
                book=prop_box(f'PropBook.{idx}.{n}',(x,y-dy),(w,0.62,0.11),mats['warm'],z=0.20+0.04*n); objs.append(book)
        elif kind=='alarm':
            clock=prop_box(f'PropAlarm.{idx}',(x,y),(1.25,0.78,0.28),mats['device']); objs.append(clock)
            face=make_ellipse(f'PropAlarmFace.{idx}',(W/2+x*PX,H/2-y*PX),0.35,0.28,mats['paper'],z=0.34,n=48); objs.append(face)
        elif kind=='fridge':
            fridge=prop_box(f'PropFridge.{idx}',(x,y),(1.55,2.55,0.50),mats['paper']); objs.append(fridge)
            handle=prop_box(f'PropFridgeHandle.{idx}',(x+0.52,y),(0.08,0.95,0.08),mats['accent'],z=0.40); objs.append(handle)
        elif kind=='shoes':
            for n,dx in enumerate((-0.42,0.42)):
                shoe=prop_box(f'PropShoe.{idx}.{n}',(x+dx,y),(0.62,0.30,0.10),mats['device']); objs.append(shoe)
        for obj in objs:
            # Keep props outside the character's readable silhouette and reveal them only
            # during their owning shot.
            if obj not in (stage,trim,shadow):
                key_visibility(obj,start,end,1.0)
    return {'stage':'desk_stage','props_per_shot':[action_rows[i].get('action') if i < len(action_rows) else beats[i][0] for i in range(len(beats))]}

def add_micro_performance_accents(arm, face, beats, rows):
    """Small, mesh-safe acting beats between major pose keys."""
    scene=bpy.context.scene
    for idx,(beat,start,end) in enumerate(beats):
        span=max(12,end-start)
        marks=(0.27,0.43,0.58,0.72)
        profiles={
            'setup': ((7,0,2.4),(0,0,-1.2),(-4,0,0.6),(0,0,0)),
            'escalation': ((12,-3,3.0),(5,0,-2.0),(-10,3,1.4),(0,0,0)),
            'reaction': ((-10,2,-2.4),(-4,0,1.6),(7,0,-0.9),(0,0,0)),
            'punchline': ((8,0,1.8),(2,0,-1.0),(-5,0,0.5),(0,0,0)),
        }[beat]
        root=arm.pose.bones['root']; head=arm.pose.bones['head']; accessory=arm.pose.bones.get('backpack')
        for frac,(dx,dy,rot) in zip(marks,profiles):
            f=min(end-2,max(start+2,round(start+span*frac))); scene.frame_set(f)
            root.location.x += dx/PX; root.location.y += dy/PX; root.rotation_euler.z += math.radians(rot)
            root.keyframe_insert('location',frame=f); root.keyframe_insert('rotation_euler',index=2,frame=f)
            head.rotation_euler.z += math.radians(-rot*0.70 + (2.0 if beat=='reaction' and frac<0.5 else 0.0))
            head.location.x -= dx/PX*0.18
            head.keyframe_insert('rotation_euler',index=2,frame=f); head.keyframe_insert('location',frame=f)
            if accessory is not None:
                lag={0.27:1.0,0.43:-0.72,0.58:0.46,0.72:-0.18}[frac]
                accessory.rotation_euler.z += math.radians(5.0*lag)
                accessory.location.x -= dx/PX*0.14; accessory.location.y -= dy/PX*0.10
                accessory.keyframe_insert('rotation_euler',frame=min(end,f+2)); accessory.keyframe_insert('location',frame=min(end,f+2))


def build(frames=190, voice_json=None, character='PIP', profile_path=None, background_path=None):
 global ASSET,BASE,TARGETS,FACE_PROFILE,CHARACTER,PROFILE
 PROFILE,BASE,TARGETS,FACE_PROFILE=load_profile(character,profile_path); CHARACTER=character; ASSET=profile_asset_dir(PROFILE)
 clear(); scene=bpy.context.scene; BEATS=build_beats(frames,voice_json); scene.render.engine='BLENDER_EEVEE'; scene.render.resolution_x=W; scene.render.resolution_y=H; scene.render.resolution_percentage=int(os.environ.get('NOVA_BLENDER_RESOLUTION','35')); scene.render.fps=24; scene.world.color=(0.012,0.018,0.026); scene.view_settings.view_transform='Standard'; scene.view_settings.look='None'; scene.view_settings.exposure=0; scene.view_settings.gamma=1
 # background
 bgmesh=bpy.data.meshes.new('BackgroundMesh'); bgmesh.from_pydata([(-WORLD_W*2,-WORLD_H*2,-.3),(WORLD_W*2,-WORLD_H*2,-.3),(WORLD_W*2,WORLD_H*2,-.3),(-WORLD_W*2,WORLD_H*2,-.3)],[],[(0,1,2,3)]); bgmesh.update(); uv=bgmesh.uv_layers.new(name='UVMap');
 for poly in bgmesh.polygons:
  for li,uvv in zip(poly.loop_indices,[(0,0),(1,0),(1,1),(0,1)]): uv.data[li].uv=uvv
 bg=bpy.data.objects.new('Background',bgmesh); bpy.context.collection.objects.link(bg); bg_mat=color_material('BG_Procedural',(0.76,0.86,0.94)); bg.data.materials.append(bg_mat)
 # Subtle lighter wall panel makes the environment feel intentional without relying on texture upload.
 panel=prop_box('WorldLightPanel',(0.0,1.30),(13.0,5.0,0.08),color_material('WorldPanel',(0.86,0.92,0.97)),z=-0.24)
 panel.keyframe_insert('location',frame=1); panel.keyframe_insert('location',frame=int(scene.frame_end))
 arm=create_armature()
 for part,z,segx,segy in [('torso',0.0,28,34),('backpack',0.02,20,32),('left_arm',0.04,24,28),('right_arm',0.05,24,28),('head',0.10,32,34)]:
  filename=PROFILE['layers'].get(part,f'{character.lower()}_{part}.png'); obj=make_grid(part,ASSET/filename,segx,segy,z); bind(obj,arm,part)
 face=face_rig(arm)
 rows=load_voice_rows(voice_json)
 world_meta=build_story_world(rows,BEATS)
 prev=pose_for_index('setup',0,rows)
 for idx,(beat,start,end) in enumerate(BEATS):
  target=pose_for_index(beat,idx,rows)
  action=(rows[idx].get('action') if idx < len(rows) else None) or beat
  chain=action_pose_chain(prev,target,beat,action,start,end)
  for frame,spec,phase in chain:
   spec=dict(spec); spec['_beat']=beat; spec['_phase']=phase; spec['_action']=action; write_pose(arm,frame,spec); key_face(face,frame,spec)
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
 # Multi-blink + eye-dart acting: the eyes react before the body fully catches up.
 for idx,(beat,start,end) in enumerate(BEATS):
  action=(rows[idx].get('action') if idx < len(rows) else '') or beat
  blink_fracs=(0.28,0.66) if action in ('shock','overwhelmed') else (0.34,0.78)
  for j,frac in enumerate(blink_fracs):
   f=start+round((end-start)*frac)
   if start+3 < f < end-2:
    for eye,pupil in zip(face['eyes'],face['pupils']):
     eye.scale.y=.12; eye.keyframe_insert('scale',index=1,frame=f); eye.scale.y=1.0; eye.keyframe_insert('scale',index=1,frame=f+2)
     if pupil is not None:
      pupil.scale.y=.07; pupil.keyframe_insert('scale',index=1,frame=f); pupil.scale.y=1.0; pupil.keyframe_insert('scale',index=1,frame=f+2)
  # Action-specific eye asymmetry and a small gaze dart.
  gaze_sign={'give_homework':1,'press_button':1,'shock':-1,'overwhelmed':1,'deadpan':0,'tap_alarm':-1,'open_fridge':1,'tap_shoes':1}.get(action,0)
  f=start+round((end-start)*0.42)
  if start < f < end:
   for side,(eye,sign) in enumerate(zip(face['eyes'],(-1,1))):
    eye.scale.x=1.0 + (0.045 if action=='shock' else -0.035 if action=='deadpan' else 0.018*sign)
    eye.scale.y=1.0 + (0.10 if action=='shock' else -0.08 if action=='overwhelmed' else 0.0)
    eye.location.x += gaze_sign*0.012*sign; eye.location.y += (-0.010 if action=='deadpan' else 0.008 if action=='shock' else 0.0)
    eye.keyframe_insert('scale',frame=f); eye.keyframe_insert('location',frame=f)
  # Multiple eye darts create readable thought/reaction beats instead of a frozen stare.
  eye_dart_fracs=(0.16,0.39,0.68)
  for dart_i,frac in enumerate(eye_dart_fracs):
   df=start+round((end-start)*frac)
   for eye,sign in zip(face['eyes'],(-1,1)):
    dart_mag={'shock':0.032,'overwhelmed':0.040,'deadpan':0.020}.get(action,0.024)
    eye.location.x += dart_mag*(1.0 if dart_i%2==0 else -0.7)*sign
    eye.keyframe_insert('location',frame=df)
  # Tiny speech-driven head bob without breaking the action pose.
  activity=(rows[idx].get('activity') if idx < len(rows) else []) if rows else []
  if activity:
   for f in range(start,end+1,4):
    q=(f-start)/max(1,end-start); ai=min(len(activity)-1,int(round(q*(len(activity)-1))))
    try: energy=max(0.0,min(1.0,float(activity[ai])))
    except (TypeError,ValueError): energy=0.0
    head=arm.pose.bones['head']; head.location.y += math.sin(f*0.42)*(0.012+0.024*energy); head.keyframe_insert('location',frame=f)
 add_micro_performance_accents(arm,face,BEATS,rows)
 bpy.ops.object.camera_add(location=(0,0,30)); cam=bpy.context.object; cam.name=f'{CHARACTER}_Camera'; cam.data.type='ORTHO'; cam.data.ortho_scale=WORLD_H; scene.camera=cam
 camera_performance_keys(arm,cam,BEATS,rows)
 # Blender's keyed transforms use its default BEZIER interpolation, providing
 # real continuous in-betweens between performance poses.
 scene.frame_start=1; scene.frame_end=frames; scene.render.image_settings.file_format='PNG'; scene.render.filepath=str(OUTDIR/'frame_####')
 effective_rows=load_voice_rows(voice_json); effective_shot_durations=[round(max(1.0,float(r.get('duration',0.0) or 0.0))+tail,3) for r,tail in zip(effective_rows[:4],(0.48,0.58,0.60,0.82))] if rows else [round((b-a)/24,3) for _,a,b in BEATS]
 meta={'engine':'blender-cutout-performance-rig-v5','character':CHARACTER,'beats':BEATS,'shots':[{'beat':beat,'frames':max(1,end-start+1),'duration':round(max(1,end-start+1)/24.0,3)} for beat,start,end in BEATS],'voice_json':str(voice_json) if voice_json else None,'effective_shot_durations':effective_shot_durations,'actions':[r.get('action') for r in effective_rows[:4]],'resolution':[W,H],'fps':24,'frames':frames,'deformable_layers':['torso','head','left_arm','right_arm','backpack'],'armature_bones':['root','head','arm.L','forearm.L','hand.L','arm.R','forearm.R','hand.R','backpack'],'facial_controls':['blink','gaze','mouth_shape','speech_micro_motion','organic_face_patches'],'pose_library':list(PROFILE.get('_pose_library',{}).keys()),'pose_map':dict(PROFILE.get('_pose_map',{})),'in_between':'action-chain-anticipation-contact-overshoot-recoil-settle','cinematic_camera':'phase-driven-hook-reveal-push-pull-impact-shake','facial_acting':['multi-blink','eye-dart','asymmetry','speech-head-bob'],'world_stage':world_meta,'action_chain':['approach','anticipation','contact_approach','contact','overshoot','recoil','settle'],'source':'character-profile-cutout-assets'}
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
