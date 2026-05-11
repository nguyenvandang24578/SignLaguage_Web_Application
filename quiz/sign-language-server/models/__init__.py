from .sta_gcn import sta_gcn_joint, sta_gcn_bone, sta_gcn_joint_motion, sta_gcn_bone_motion, sta_gcn_joint_wlasl100
from .ctrgcn import ctrgcn_joint, ctrgcn_bone, ctrgcn_joint_motion, ctrgcn_bone_motion, ctrgcn_joint_wlasl100
from .tdgcn import tdgcn_joint, tdgcn_bone, tdgcn_joint_motion, tdgcn_bone_motion, tdgcn_joint_wlasl100
model_dict = {
    'tdgcn_joint': tdgcn_joint,
    'ctrgcn_joint_wlasl_100':ctrgcn_joint_wlasl100,
    'sta_gcn_joint_wlasl':sta_gcn_joint_wlasl100,
    'tdgcn_joint_wlasl_100': tdgcn_joint_wlasl100,
    'tdgcn_bone': tdgcn_bone,
    'tdgcn_joint_motion': tdgcn_joint_motion,
    'tdgcn_bone_motion': tdgcn_bone_motion,
    'ctrgcn_joint': ctrgcn_joint,
    'ctrgcn_bone': ctrgcn_bone,
    'ctrgcn_joint_motion': ctrgcn_joint_motion,
    'ctrgcn_bone_motion': ctrgcn_bone_motion,
    'sta_gcn_joint': sta_gcn_joint, 
    'sta_gcn_bone': sta_gcn_bone, 
    'sta_gcn_joint_motion': sta_gcn_joint_motion, 
    'sta_gcn_bone_motion': sta_gcn_bone_motion, 
}
