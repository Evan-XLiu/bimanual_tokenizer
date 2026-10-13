"""Build a reproducible simulation overlay; never edit the pinned source assets."""
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from lower_body_policy import ROOT, BODY_NAMES, load_candidate

sys.path.insert(0, str(ROOT/'.local/unitree_g1/sim_deps'))
import mujoco


def floats(text):
    return np.fromstring(text, sep=' ')


def inertial(link):
    """URDF inertia about its CoM, rotated into body frame for MJCF fullinertia."""
    item = link.find('inertial')
    origin = item.find('origin')
    i = item.find('inertia').attrib
    matrix = np.array([[float(i['ixx']), float(i['ixy']), float(i['ixz'])],
                       [float(i['ixy']), float(i['iyy']), float(i['iyz'])],
                       [float(i['ixz']), float(i['iyz']), float(i['izz'])]])
    r = Rotation.from_euler('xyz', floats(origin.get('rpy', '0 0 0'))).as_matrix()
    matrix = r @ matrix @ r.T
    if np.linalg.eigvalsh(matrix).min() <= 0:
        raise ValueError('Nonpositive inertia')
    values = [matrix[0, 0], matrix[1, 1], matrix[2, 2], matrix[0, 1], matrix[0, 2], matrix[1, 2]]
    return ET.Element('inertial', pos=origin.get('xyz'), mass=item.find('mass').get('value'),
                      fullinertia=' '.join(map(str, values)))


def dex_source():
    lock = json.loads((ROOT/'artifacts/unitree_g1/dex1_model.lock.json').read_text())
    root = ROOT/lock['source_root']
    for name, digest in lock['sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Dex1 source hash mismatch: '+name)
    return root, lock


def build_model(model_name='29dof', dt=.001, servo='implicit', dex1=False,
                extra_mass=0., payload_mass=0.):
    if model_name not in ('12dof', '29dof') or servo not in ('implicit', 'explicit'):
        raise ValueError('Unknown simulation model/servo')
    if not all(math.isfinite(x) and x >= 0 for x in (extra_mass, payload_mass)):
        raise ValueError('Invalid additional mass')
    if model_name != '29dof' and (dex1 or extra_mass or payload_mass):
        raise ValueError('Hand overlays require 29dof')
    root, c, lock = load_candidate()
    source = root/'resources/robots/g1_description'
    path = source/('g1_12dof.xml' if model_name == '12dof' else 'g1_29dof_rev_1_0.xml')
    xml = ET.parse(path).getroot()
    xml.find('compiler').set('meshdir', str(source/'meshes'))
    if model_name == '12dof':
        ET.SubElement(xml.find('worldbody'), 'geom', name='floor', type='plane', size='0 0 0.05')
    for mesh in xml.findall('asset/mesh'):
        mesh.set('file', str(source/'meshes'/mesh.get('file')))
    dex_meta = None
    if dex1:
        dexroot, dexlock = dex_source()
        urdf = ET.parse(dexroot/'robots/g1_description/g1_29dof_mode_15_with_dex1_1.urdf').getroot()
        links = {e.get('name'): e for e in urdf.findall('link')}
        joints = {e.get('name'): e for e in urdf.findall('joint')}
        body_urdf = ET.parse(source/'g1_29dof_rev_1_0.urdf').getroot()
        original_links = {e.get('name'): e for e in body_urdf.findall('link')}
        assets = xml.find('asset')
        equality = ET.SubElement(xml, 'equality')
        for side in ('left', 'right'):
            wrist = xml.find(f'.//body[@name="{side}_wrist_yaw_link"]')
            # MJCF fused rubber-hand mass into wrist. Restore bare wrist inertia
            # from matching URDF before adding Dex1: avoid double-counting hands.
            wrist.remove(wrist.find('inertial'))
            wrist.insert(0, inertial(original_links[f'{side}_wrist_yaw_link']))
            for geom in list(wrist.findall('geom')):
                if geom.get('mesh') == f'{side}_rubber_hand':
                    wrist.remove(geom)
            mount = joints[f'{side}_base_joint'].find('origin')
            quat = Rotation.from_euler('xyz', floats(mount.get('rpy'))).as_quat()[[3, 0, 1, 2]]
            base = ET.SubElement(wrist, 'body', name=f'{side}_dex1_base_link',
                                 pos=mount.get('xyz'), quat=' '.join(map(str, quat)))
            for index in (0, 1, 2):
                name = f'{side}_dex1_base_link' if index == 0 else f'{side}_dex1_finger_link_{index}'
                body = base if index == 0 else ET.SubElement(base, 'body', name=name)
                link = links[name]
                body.append(inertial(link))
                if index:
                    joint = joints[f'{side}_dex1_finger_joint_{index}']
                    limit = joint.find('limit')
                    ET.SubElement(body, 'joint', name=joint.get('name'), type='slide',
                                  axis=joint.find('axis').get('xyz'),
                                  range=limit.get('lower')+' '+limit.get('upper'), damping='1')
                    ET.SubElement(xml.find('actuator'), 'position', name=joint.get('name'),
                                  joint=joint.get('name'), kp='500', kv='10',
                                  forcelimited='true', forcerange='-20 20')
                for role in ('visual', 'collision'):
                    mesh_file = link.find(f'{role}/geometry/mesh').get('filename')
                    mesh_name = name+'_'+role
                    ET.SubElement(assets, 'mesh', name=mesh_name,
                                  file=str(dexroot/'robots/g1_description'/mesh_file))
                    ET.SubElement(body, 'geom', type='mesh', mesh=mesh_name,
                                  contype='0' if role == 'visual' else '1',
                                  conaffinity='0' if role == 'visual' else '1',
                                  density='0', group='1' if role == 'visual' else '0')
            ET.SubElement(equality, 'joint', joint1=f'{side}_dex1_finger_joint_1',
                          joint2=f'{side}_dex1_finger_joint_2', polycoef='0 1 0 0 0')
        dex_meta = dict(commit=dexlock['commit'], mass_per_hand_kg=sum(float(links[f'left_dex1_{s}'].find('inertial/mass').get('value'))
                        for s in ('base_link', 'finger_link_1', 'finger_link_2')),
                        mount='mode15 wrist +0.0415m X assumed on mode5; not measured',
                        finger_drive='URDF prismatic meters; no calibrated lowcmd motor-to-slider conversion')
    for side in ('left', 'right'):
        wrist = xml.find(f'.//body[@name="{side}_wrist_yaw_link"]')
        for name, mass, pos in [('mount', extra_mass, '.04 0 0'),
                                ('payload', payload_mass if side == 'left' else 0., '.16 0 0')]:
            if mass:
                b = ET.SubElement(wrist, 'body', name=f'{side}_{name}_sensitivity', pos=pos)
                ET.SubElement(b, 'inertial', pos='0 0 0', mass=str(mass),
                              diaginertia=' '.join([str(mass*.03**2*2/5)]*3))
    m = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))
    m.opt.timestep = dt
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST if servo == 'implicit' else mujoco.mjtIntegrator.mjINT_EULER
    count = 12 if model_name == '12dof' else 29
    kp = np.array(c['kps'] + (c['arm_waist_kps'] if count == 29 else []), dtype=float)
    kd = np.array(c['kds'] + (c['arm_waist_kds'] if count == 29 else []), dtype=float)
    for i, name in enumerate(BODY_NAMES[:count]):
        a = m.actuator(name).id
        if servo == 'implicit':
            # Identical continuous-time PD law, engine-visible velocity derivative.
            m.actuator_gaintype[a] = mujoco.mjtGain.mjGAIN_FIXED
            m.actuator_gainprm[a, 0] = kp[i]
            m.actuator_biastype[a] = mujoco.mjtBias.mjBIAS_AFFINE
            m.actuator_biasprm[a, :3] = [0., -kp[i], -kd[i]]
    return m, c, dict(source_commit=lock['commit'], servo=servo, dex1=dex_meta,
                      extra_mass_each_wrist_kg=extra_mass, left_payload_kg=payload_mass,
                      total_mass_kg=float(m.body_mass.sum()))
