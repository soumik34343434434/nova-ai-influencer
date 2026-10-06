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
ASSET = ROOT / 'artifacts' / 'cartoon' / 'blender' / 'pip_cutout_layers'
OUTDIR = ROOT / 'artifacts' / 'cartoon' / 'blender' / 'pip_cutout_rig'
BLEND = ROOT / 'artifacts' / 'cartoon' / 'blender' / 'pip_cutout_rig_v2.blend'
W,H=1080,1920
PX=108.0
WORLD_H=H/PX
WORLD_W=W/PX

BASE={
 'root':(550,1260),'neck':(545,1035),'head_center':(575,815),
 'left_shoulder':(390,1095),'left_elbow':(305,1125),'left_wrist':(275,1205),
 'right_shoulder':(770,1080),'right_elbow':(845,1095),'right_wrist':(875,1160),
 'head_rot':0.0,'root_rot':0.0,'eye_open':1.0,'gaze':(0.10,0.0),'mouth':'smile'
}
TARGETS={
 'setup':BASE,
 'escalation':{**BASE,'root':(590,1250),'neck':(590,1015),'head_center':(600,800),
  'left_shoulder':(410,1080),'left_elbow':(330,1000),'left_wrist':(300,925),
  'right_shoulder':(805,1065),'right_elbow':(890,990),'right_wrist':(930,915),
  'head_rot':8.0,'root_rot':3.5,'eye_open':1.18,'gaze':(0.48,-0.05),'mouth':'open'},
 'reaction':{**BASE,'root':(500,1345),'neck':(495,1110),'head_center':(490,900),
  'left_shoulder':(355,1180),'left_elbow':(275,1255),'left_wrist':(245,1330),
  'right_shoulder':(700,1165),'right_elbow':(775,1240),'right_wrist':(810,1310),
  'head_rot':-12.0,'root_rot':-5.0,'eye_open':0.92,'gaze':(0.36,0.02),'mouth':'frown'},
 'punchline':{**BASE,'root':(575,1270),'neck':(570,1045),'head_center':(570,835),
  'left_shoulder':(415,1100),'left_elbow':(395,1165),'left_wrist':(385,1220),
  'right_shoulder':(775,1095),'right_elbow':(800,1165),'right_wrist':(815,1220),
  'head_rot':1.5,'root_rot':0.5,'eye_open':0.88,'gaze':(0.0,0.0),'mouth':'flat'}
}
BEATS=[('setup',1,42),('escalation',43,88),('reaction',89,134),('punchline',135,190)]


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
 bpy.ops.object.armature_add(enter_editmode=True,location=(0,0,0)); arm=bpy.context.object; arm.name='PIP_Rig'; eb=arm.data.edit_bones; eb.remove(eb[0]); bones={}
 def add(name,head,tail,parent=None,connect=False):
  b=eb.new(name); b.head=px(head); b.tail=px(tail); b.parent=bones.get(parent); b.use_connect=connect; bones[name]=b
 add('root',BASE['root'],BASE['neck']); add('head',BASE['neck'],BASE['head_center'],'root');
 add('arm.L',BASE['left_shoulder'],BASE['left_elbow'],'root'); add('forearm.L',BASE['left_elbow'],BASE['left_wrist'],'arm.L',True)
 add('arm.R',BASE['right_shoulder'],BASE['right_elbow'],'root'); add('forearm.R',BASE['right_elbow'],BASE['right_wrist'],'arm.R',True)
 add('backpack',(250,1135),(250,1000),'root')
 bpy.ops.object.mode_set(mode='POSE')
 for b in arm.pose.bones: b.rotation_mode='XYZ'
 bpy.ops.object.mode_set(mode='OBJECT'); return arm


def distance_weight(p,a,b):
 pa,pb=px(a),px(b); ab=pb-pa; t=max(0,min(1,(p-pa).dot(ab)/max(1e-8,ab.length_squared))); return 1-t,t


def bind(obj,arm,part):
 names=('root','head','arm.L','forearm.L','arm.R','forearm.R','backpack')
 groups={n:obj.vertex_groups.new(name=n) for n in names}
 for v in obj.data.vertices:
  p=Vector((v.co.x,v.co.y,0)); pp=(p.x*PX+W/2,H/2-p.y*PX)
  if part=='head': groups['head'].add([v.index],1,'REPLACE')
  elif part=='backpack': groups['backpack'].add([v.index],1,'REPLACE')
  elif part=='left_arm':
   # Stable first-pass deformation: the complete arm/hand silhouette follows the
   # shoulder bone. This avoids UV tearing while keeping the limb physically
   # attached to the torso; forearm split is a later refinement.
   groups['arm.L'].add([v.index],1,'REPLACE')
  elif part=='right_arm':
   groups['arm.R'].add([v.index],1,'REPLACE')
  else: groups['root'].add([v.index],1,'REPLACE')
 mod=obj.modifiers.new('PIP_Armature_Deform','ARMATURE'); mod.object=arm; mod.use_deform_preserve_volume=False


def bone_parent_keep_world(obj,arm,bone):
 world=obj.matrix_world.copy(); obj.parent=arm; obj.parent_type='BONE'; obj.parent_bone=bone; obj.matrix_world=world


def make_ellipse(name,center,rx,ry,mat,z=.4,n=48):
 c=px(center); pts=[(c.x+rx*math.cos(math.tau*i/n),c.y+ry*math.sin(math.tau*i/n)) for i in range(n)]; cx=sum(x for x,y in pts)/n; cy=sum(y for x,y in pts)/n
 verts=[(x-cx,y-cy,z) for x,y in pts]; mesh=bpy.data.meshes.new(name+'Mesh'); mesh.from_pydata(verts,[],[tuple(range(n))]); mesh.update(); obj=bpy.data.objects.new(name,mesh); bpy.context.collection.objects.link(obj); obj.location=(cx,cy,0); obj.data.materials.append(mat); return obj


def face_rig(arm):
 dark=color_material('FaceScreen',(0.025,0.045,0.065)); white=color_material('FaceLED',(0.95,1.0,1.0)); black=color_material('FacePupil',(0.01,0.015,0.02))
 screen=make_ellipse('FaceScreen',(585,815),1.82,1.22,dark,.31)
 eye_l=make_ellipse('Eye.L',(500,805),.34,.42,white,.35); eye_r=make_ellipse('Eye.R',(650,805),.34,.42,white,.35)
 pup_l=make_ellipse('Pupil.L',(500,805),.11,.145,black,.37); pup_r=make_ellipse('Pupil.R',(650,805),.11,.145,black,.37)
 mouth=make_ellipse('Mouth',(585,875),.56,.13,white,.36)
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
  p1.rotation_euler.z=ta-base_a; p2.rotation_euler.z=(tb-ta)-(base_b-base_a); p1.keyframe_insert('rotation_euler',index=2,frame=frame); p2.keyframe_insert('rotation_euler',index=2,frame=frame)
 head=arm.pose.bones['head']; head.rotation_euler.z=math.radians(spec['head_rot']); head.keyframe_insert('rotation_euler',index=2,frame=frame)
 scale=1.0+({'setup':0,'escalation':0.035,'reaction':-0.035,'punchline':-0.01}.get(spec.get('_beat','setup'),0)); root.scale=(1/scale,scale,1); root.keyframe_insert('scale',frame=frame)
 bp=arm.pose.bones['backpack']; bp.rotation_euler.z=math.radians(spec['root_rot']*.45); bp.keyframe_insert('rotation_euler',index=2,frame=frame)


def key_face(face,frame,spec):
 c=px(spec['head_center']); base_c=px(BASE['head_center']); a=math.radians(spec['head_rot']); co,si=math.cos(a),math.sin(a)
 offsets={'screen':Vector((px((585,815)).x-base_c.x,px((585,815)).y-base_c.y,0)),'Eye.L':px((500,805))-base_c,'Eye.R':px((650,805))-base_c,'Pupil.L':px((500,805))-base_c,'Pupil.R':px((650,805))-base_c,'Mouth':px((585,875))-base_c}
 def world_pos(v): return (c.x + co*v.x-si*v.y, c.y + si*v.x+co*v.y, 0)
 for name,obj in [('screen',face['screen']),('Eye.L',face['eyes'][0]),('Eye.R',face['eyes'][1]),('Pupil.L',face['pupils'][0]),('Pupil.R',face['pupils'][1]),('Mouth',face['mouth'])]:
  obj.location=world_pos(offsets[name]); obj.rotation_euler.z=a; obj.keyframe_insert('location',frame=frame); obj.keyframe_insert('rotation_euler',index=2,frame=frame)
 for eye,pupil,side in [(face['eyes'][0],face['pupils'][0],-1),(face['eyes'][1],face['pupils'][1],1)]:
  eye.scale.y=spec['eye_open']; pupil.scale.y=max(.12,spec['eye_open']); pupil.location.x=world_pos(offsets['Pupil.L' if side<0 else 'Pupil.R'])[0]+spec['gaze'][0]*.18; pupil.location.y=world_pos(offsets['Pupil.L' if side<0 else 'Pupil.R'])[1]+spec['gaze'][1]*.14; eye.keyframe_insert('scale',index=1,frame=frame); pupil.keyframe_insert('scale',index=1,frame=frame); pupil.keyframe_insert('location',frame=frame)
 mouth=face['mouth']; mouth.scale={'smile':(1,1,1),'open':(.90,1.70,1),'frown':(.88,.60,1),'flat':(.82,.30,1)}[spec['mouth']]; mouth.keyframe_insert('scale',frame=frame)


def build(frames=190):
 clear(); scene=bpy.context.scene; scene.render.engine='BLENDER_EEVEE'; scene.render.resolution_x=W; scene.render.resolution_y=H; scene.render.resolution_percentage=int(os.environ.get('NOVA_BLENDER_RESOLUTION','35')); scene.render.fps=24; scene.world.color=(0.0,0.0,0.0); scene.view_settings.view_transform='Standard'; scene.view_settings.look='None'; scene.view_settings.exposure=0; scene.view_settings.gamma=1
 # background
 bgmesh=bpy.data.meshes.new('BackgroundMesh'); bgmesh.from_pydata([(-WORLD_W/2,-WORLD_H/2,-.3),(WORLD_W/2,-WORLD_H/2,-.3),(WORLD_W/2,WORLD_H/2,-.3),(-WORLD_W/2,WORLD_H/2,-.3)],[],[(0,1,2,3)]); bgmesh.update(); uv=bgmesh.uv_layers.new(name='UVMap');
 for poly in bgmesh.polygons:
  for li,uvv in zip(poly.loop_indices,[(0,0),(1,0),(1,1),(0,1)]): uv.data[li].uv=uvv
 bg=bpy.data.objects.new('Background',bgmesh); bpy.context.collection.objects.link(bg); bg.data.materials.append(image_material('BG',ASSET/'pip_background.jpg'))
 arm=create_armature()
 for part,z,segx,segy in [('torso',0.0,28,34),('backpack',0.02,20,32),('left_arm',0.04,24,28),('right_arm',0.05,24,28),('head',0.10,32,34)]:
  obj=make_grid(part,ASSET/f'pip_{part}.png',segx,segy,z); bind(obj,arm,part)
 face=face_rig(arm)
 prev=TARGETS['setup']
 for beat,start,end in BEATS:
  target=TARGETS[beat]
  dur=end-start
  specs=[(start,pose_spec(prev,target,-0.12)),(start+round(dur*.48),pose_spec(prev,target,1.10)),(start+round(dur*.72),target),(end,target)]
  for frame,spec in specs:
   spec=dict(spec); spec['_beat']=beat; write_pose(arm,frame,spec); key_face(face,frame,spec)
  prev=target
 # Add speech-reactive mouth shape cues: fast micro-open/close around each phrase center.
 for beat,start,end in BEATS:
  spec=TARGETS[beat]; mid=start+round((end-start)*.45)
  for f,sy in [(mid-2,.55),(mid,.95),(mid+2,.38),(mid+4,1.0)]:
   face['mouth'].scale.y={'smile':1,'open':1.5,'frown':.7,'flat':.3}[spec['mouth']]*sy; face['mouth'].keyframe_insert('scale',frame=f)
 # Blink on non-impact moments.
 for beat,start,end in BEATS:
  f=start+round((end-start)*.34)
  for eye,pupil in zip(face['eyes'],face['pupils']):
   eye.scale.y=.10; pupil.scale.y=.06; eye.keyframe_insert('scale',index=1,frame=f); pupil.keyframe_insert('scale',index=1,frame=f); eye.scale.y=1; pupil.scale.y=1; eye.keyframe_insert('scale',index=1,frame=f+2); pupil.keyframe_insert('scale',index=1,frame=f+2)
 bpy.ops.object.camera_add(location=(0,0,30)); cam=bpy.context.object; cam.name='PIP_Camera'; cam.data.type='ORTHO'; cam.data.ortho_scale=WORLD_H; scene.camera=cam
 scene.frame_start=1; scene.frame_end=frames; scene.render.image_settings.file_format='PNG'; scene.render.filepath=str(OUTDIR/'frame_####')
 meta={'engine':'blender-pip-cutout-rig-v2','resolution':[W,H],'fps':24,'frames':frames,'deformable_layers':['torso','head','left_arm','right_arm','backpack'],'armature_bones':['root','head','arm.L','forearm.L','arm.R','forearm.R','backpack'],'facial_controls':['blink','gaze','mouth_shape','speech_micro_motion'],'in_between':'overshoot-and-settle-bezier','source':'clean-pip-master-cutout'}
 t=bpy.data.texts.new('NOVA_BLENDER_RIG_METADATA.json'); t.write(json.dumps(meta,indent=2)); return scene,meta


def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--build',action='store_true'); ap.add_argument('--render',action='store_true'); ap.add_argument('--frames',type=int,default=190); ap.add_argument('--blend-output',type=Path,default=BLEND); args=ap.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
 OUTDIR.mkdir(parents=True,exist_ok=True); scene,meta=build(args.frames); bpy.ops.wm.save_as_mainfile(filepath=str(args.blend_output))
 if args.render:
  selected=os.environ.get('NOVA_KEY_FRAMES','').strip()
  if selected:
   for token in selected.split(','):
    f=int(token.strip()); scene.frame_set(f); scene.render.filepath=str(OUTDIR/f'key_{f:04d}.png'); bpy.ops.render.render(write_still=True)
  else:
   bpy.ops.render.render(animation=True)
if __name__=='__main__': main()
