import os
import io

import json
import trimesh
import random
from tqdm import tqdm

import matplotlib.pyplot as plt
import transforms3d
import pyrender
import numpy as np
import cv2
import torch

import psutil
import os.path as osp
import pickle

def network_grad_ratio(model):
    '''
    for debug
    :return:
    '''
    gradsum = 0
    gradmax = 0
    datasum = 0
    layercnt = 0
    for param in model.parameters():
        if param.grad is not None:
            if param.grad.abs().max() > gradmax:
                gradmax = param.grad.abs().max()
            grad = param.grad.abs().mean()
            data = param.data.abs().mean()
            # print(grad)
            gradsum += grad
            datasum += data
            layercnt += 1
    gradsum /= layercnt
    datasum /= layercnt
    return float(gradmax), float(gradsum), float(datasum)
    
def load_bop_meshes(model_path):
    # load meshes
    meshFiles = [f for f in os.listdir(model_path) if f.endswith('.ply')]
    meshFiles.sort()
    meshes = []
    objID_2_clsID = {}
    for i in range(len(meshFiles)):
        mFile = meshFiles[i]
        objId = int(os.path.splitext(mFile)[0][4:])
        objID_2_clsID[str(objId)] = i
        meshes.append(trimesh.load(model_path + mFile))
        # print('mesh from "%s" is loaded' % (model_path + mFile))
    # 
    return meshes, objID_2_clsID

def load_bbox_3d(jsonFile):
    with open(jsonFile, 'r') as f:
        bbox_3d = json.load(f)
    return bbox_3d

def collect_mesh_bbox(meshpath, outjson, oriented=False):
    meshes, _ = load_bop_meshes(meshpath)
    allv = []
    for ms in meshes:
        if oriented:
            bbox = ms.bounding_box_oriented.vertices
        else:
            bbox = ms.bounding_box.vertices
        allv.append(bbox.tolist())
    with open(outjson, 'w') as outfile:
        json.dump(allv, outfile, indent=4)

def collect_bop_imagelist(dataDir):
    assert(dataDir[-1] == '/')

    lastSubDir = dataDir[dataDir[:-1].rfind('/')+1:]
    parentDir = dataDir[:dataDir.rfind(lastSubDir)]

    all_images = []
    sub_images = {}

    subset = os.listdir(dataDir)
    subset.sort()

    for ss in subset:
        rgbPath = dataDir + ss + '/rgb/'
        imglist = [f for f in os.listdir(rgbPath) if f.endswith('.png') or f.endswith('.jpg')]
        imglist.sort()
        if ss not in sub_images:
            sub_images[ss] = []
        for im in imglist:
            filename = lastSubDir + ss + '/rgb/' + im
            all_images.append(filename)
            sub_images[ss].append(filename)

    return all_images, sub_images

def quaternion2rotation(quat):
    assert (len(quat) == 4)
    # normalize first
    quat = quat / np.linalg.norm(quat)
    a, b, c, d = quat

    a2 = a * a
    b2 = b * b
    c2 = c * c
    d2 = d * d
    ab = a * b
    ac = a * c
    ad = a * d
    bc = b * c
    bd = b * d
    cd = c * d

    # s = a2 + b2 + c2 + d2

    m0 = a2 + b2 - c2 - d2
    m1 = 2 * (bc - ad)
    m2 = 2 * (bd + ac)
    m3 = 2 * (bc + ad)
    m4 = a2 - b2 + c2 - d2
    m5 = 2 * (cd - ab)
    m6 = 2 * (bd - ac)
    m7 = 2 * (cd + ab)
    m8 = a2 - b2 - c2 + d2

    return np.array([m0, m1, m2, m3, m4, m5, m6, m7, m8]).reshape(3, 3)

def rotation2quaternion(M):
    tr = np.trace(M)
    m = M.reshape(-1)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[7] - m[5]) / s
        y = (m[2] - m[6]) / s
        z = (m[3] - m[1]) / s
    elif m[0] > m[4] and m[0] > m[8]:
        s = np.sqrt(1.0 + m[0] - m[4] - m[8]) * 2
        w = (m[7] - m[5]) / s
        x = 0.25 * s
        y = (m[1] + m[3]) / s
        z = (m[2] + m[6]) / s
    elif m[4] > m[8]:
        s = np.sqrt(1.0 + m[4] - m[0] - m[8]) * 2
        w = (m[2] - m[6]) / s
        x = (m[1] + m[3]) / s
        y = 0.25 * s
        z = (m[5] + m[7]) / s
    else:
        s = np.sqrt(1.0 + m[8] - m[0] - m[4]) * 2
        w = (m[3] - m[1]) / s
        x = (m[2] + m[6]) / s
        y = (m[5] + m[7]) / s
        z = 0.25 * s
    Q = np.array([w, x, y, z]).reshape(-1)
    return Q

def generate_shiftscalerotate_matrix(shift_limit, scale_limit, rotate_limit, width, height):
    dw = int(width * shift_limit)
    dh = int(height * shift_limit)
    pleft = random.randint(-dw, dw)
    ptop = random.randint(-dh, dh)
    shiftM = np.array([[1.0, 0.0, -pleft], [0.0, 1.0, -ptop], [0.0, 0.0, 1.0]])  # translation

    # random rotation and scaling
    cx = width / 2 # fix the rotation center to the image center
    cy = height / 2
    ang = random.uniform(-rotate_limit, rotate_limit)
    sfactor = random.uniform(-scale_limit, +scale_limit) + 1
    tmp = cv2.getRotationMatrix2D((cx, cy), ang, sfactor)  # rotation with scaling
    rsM = np.concatenate((tmp, [[0, 0, 1]]), axis=0)

    # combination
    M = np.matmul(rsM, shiftM)

    return M.astype(np.float32)

def distort_hsv(img, h_ratio, s_ratio, v_ratio):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # hue, sat, val
    h = img_hsv[:, :, 0].astype(np.float32)  # hue
    s = img_hsv[:, :, 1].astype(np.float32)  # saturation
    v = img_hsv[:, :, 2].astype(np.float32)  # value
    a = random.uniform(-1, 1) * h_ratio + 1
    b = random.uniform(-1, 1) * s_ratio + 1
    c = random.uniform(-1, 1) * v_ratio + 1
    h *= a
    s *= b
    v *= c
    img_hsv[:, :, 0] = h if a < 1 else h.clip(None, 179)
    img_hsv[:, :, 1] = s if b < 1 else s.clip(None, 255)
    img_hsv[:, :, 2] = v if c < 1 else v.clip(None, 255)
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

def distort_noise(img, noise_ratio=0):
    # add noise
    noisesigma = random.uniform(0, noise_ratio)
    gauss = np.random.normal(0, noisesigma, img.shape) * 255
    img = img + gauss

    img[img > 255] = 255
    img[img < 0] = 0

    return np.uint8(img)

def distort_smooth(img, smooth_ratio=0):
    # add smooth
    smoothsigma = random.uniform(0, smooth_ratio)
    res = cv2.GaussianBlur(img, (7, 7), smoothsigma, cv2.BORDER_DEFAULT)
    return res

def get_available_memory():
    mem_stat = psutil.virtual_memory()
    return mem_stat.available / mem_stat.total

def load_image_cached(img_path, mem_cache=None):
    if mem_cache != None and not img_path in mem_cache:
        if get_available_memory() > 0.1:
            with open(img_path, 'rb') as f:
                mem_cache[img_path] = f.read()
    if mem_cache != None and img_path in mem_cache:
        return cv2.imdecode(np.fromstring(mem_cache[img_path], np.uint8), cv2.IMREAD_UNCHANGED)
    else:
        return cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

def load_json_cached(json_path, mem_cache=None):
    if mem_cache != None and not json_path in mem_cache:
        if get_available_memory() > 0.1:
            with open(json_path, 'r') as f:
                mem_cache[json_path] = f.read()
    if mem_cache != None and json_path in mem_cache:
        return json.loads(mem_cache[json_path])
    else:
        return json.load(open(json_path))
        
def get_single_bop_annotation(img_path, objID_2_clsID, mem_cache=None):
    img_path = img_path.strip()
    # 
    gt_dir, tmp, imgName = img_path.rsplit('/', 2)
    assert(tmp == 'rgb')
    imgBaseName, _ = os.path.splitext(imgName)
    # 
    camera_file = gt_dir + '/scene_camera.json'
    gt_file = gt_dir + "/scene_gt.json"
    # gt_info_file = gt_dir + "/scene_gt_info.json"
    gt_mask_visib = gt_dir + "/mask_visib/"

    gt_json = load_json_cached(gt_file, mem_cache)
    cam_json = load_json_cached(camera_file, mem_cache)
    # gt_info_json = json.load(open(gt_info_file))

    im_id_str = str(int(imgBaseName))
    if im_id_str in cam_json:
        annot_camera = cam_json[im_id_str]
    else:
        annot_camera = cam_json[imgBaseName]
    # 
    if im_id_str in gt_json:
        annot_poses = gt_json[im_id_str]
    else:
        annot_poses = gt_json[imgBaseName]
    # 
    # annot_camera = cam_json[("%06d" % im_id)]
    # annot_poses = gt_json[("%06d" % im_id)]
    # annot_infos = gt_info_json[str(im_id)]

    objCnt = len(annot_poses)
    K = np.array(annot_camera['cam_K']).reshape(3,3)

    class_ids = []
    # bbox_objs = []
    rotations = []
    translations = []
    merged_mask = None
    instance_idx = 1
    for i in range(objCnt):
        mask_vis_file = gt_mask_visib + ("%s_%06d.png" %(imgBaseName, i))
        mask_vis = load_image_cached(mask_vis_file, mem_cache)
        height = mask_vis.shape[0]
        width = mask_vis.shape[1]
        if merged_mask is None:
            merged_mask = np.zeros((height, width), np.uint8) # segmenation masks
        # 
        R = np.array(annot_poses[i]['cam_R_m2c']).reshape(3,3)
        T = np.array(annot_poses[i]['cam_t_m2c']).reshape(3,1)
        obj_id = str(annot_poses[i]['obj_id'])
        if not obj_id in objID_2_clsID:
            continue
        cls_id = objID_2_clsID[obj_id]
        # 
        # bbox_objs.append(bbox)
        class_ids.append(cls_id)
        rotations.append(R)
        translations.append(T)
        # compose segmentation labels
        merged_mask[mask_vis==255] = instance_idx
        instance_idx += 1
    
    return K, merged_mask, class_ids, rotations, translations

def draw_bounding_box(cvImg, R, T, mesh_or_bbox, intrinsics, color, bbox_trans=None):
    thickness = 2
    if isinstance(mesh_or_bbox, trimesh.Trimesh):
        bbox = mesh_or_bbox.bounding_box_oriented.vertices
    else:
        bbox = np.array(mesh_or_bbox)
    rep = np.matmul(intrinsics, np.matmul(R, bbox.T) + T)
    x = np.int32(rep[0]/rep[2] + 0.5)
    y = np.int32(rep[1]/rep[2] + 0.5)
  
    if bbox_trans is not None:
        x = rep[0]/rep[2]
        y = rep[1]/rep[2]
        v_ones = np.ones(x.shape[0])#.to(bbox_trans.device)
        kpt2d = np.stack([x, y, v_ones])
        trans_kpt2d = bbox_trans @ kpt2d
        x = np.int32(trans_kpt2d[0,...] + 0.5)
        y = np.int32(trans_kpt2d[1,...] + 0.5)
    
    bbox_lines = [0, 1, 0, 2, 0, 4, 5, 1, 5, 4, 6, 2, 6, 4, 3, 2, 3, 1, 7, 3, 7, 5, 7, 6]
    for i in range(12):
        id1 = bbox_lines[2*i]
        id2 = bbox_lines[2*i+1]
        cvImg = cv2.line(cvImg, (x[id1],y[id1]), (x[id2],y[id2]), color, thickness=thickness, lineType=cv2.LINE_AA)
    return cvImg

def draw_pose_axis(cvImg, R, T, mesh_or_bbox, intrinsics, bbox_trans=None):
    thickness = 2
    if isinstance(mesh_or_bbox, trimesh.Trimesh):
        bbox = mesh_or_bbox.bounding_box_oriented.vertices
    else:
        bbox = mesh_or_bbox
    radius = np.linalg.norm(bbox, axis=1).mean()
    aPts = np.array([[0,0,0],[0,0,radius],[0,radius,0],[radius,0,0]])
    rep = np.matmul(intrinsics, np.matmul(R, aPts.T) + T)
    x = np.int32(rep[0]/rep[2] + 0.5)
    y = np.int32(rep[1]/rep[2] + 0.5)
    # x = rep[0]/rep[2]
    # y = rep[1]/rep[2]

    if bbox_trans is not None:
        x = rep[0]/rep[2]
        y = rep[1]/rep[2]
        v_ones = np.ones(x.shape[0])#.to(bbox_trans.device)
        kpt2d = np.stack([x, y, v_ones])
        trans_kpt2d = bbox_trans @ kpt2d
        x = np.int32(trans_kpt2d[0,...] + 0.5)
        y = np.int32(trans_kpt2d[1,...] + 0.5)

    cvImg = cv2.line(cvImg, (x[0],y[0]), (x[1],y[1]), (0,0,255), thickness=thickness, lineType=cv2.LINE_AA)
    cvImg = cv2.line(cvImg, (x[0],y[0]), (x[2],y[2]), (0,255,0), thickness=thickness, lineType=cv2.LINE_AA)
    cvImg = cv2.line(cvImg, (x[0],y[0]), (x[3],y[3]), (255,0,0), thickness=thickness, lineType=cv2.LINE_AA)
    return cvImg

def draw_z_axis(cvImg, R, T, mesh_or_bbox, intrinsics, color):
    thickness = 2
    if isinstance(mesh_or_bbox, trimesh.Trimesh):
        bbox = mesh_or_bbox.bounding_box_oriented.vertices
    else:
        bbox = mesh_or_bbox
    radius = np.linalg.norm(bbox, axis=1).mean()
    aPts = np.array([[0,0,0],[0,0,radius]])
    rep = np.matmul(intrinsics, np.matmul(R, aPts.T) + T)
    x = np.int32(rep[0]/rep[2] + 0.5)
    y = np.int32(rep[1]/rep[2] + 0.5)
    cvImg = cv2.line(cvImg, (x[0],y[0]), (x[1],y[1]), color, thickness=thickness, lineType=cv2.LINE_AA)
    return cvImg

def draw_pose_contour(cvImg, R, T, mesh, K, color):
    # 
    h, w, _ = cvImg.shape
    currentpose = np.concatenate((R, T.reshape(-1, 1)), axis=-1)
    _, depth = render_objects([mesh], [0], [currentpose], K, w, h)
    validMap = (depth>0).astype(np.uint8)
    # 
    # find contour
    contours, _ = cv2.findContours(validMap, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    # cvImg = cv2.drawContours(cvImg, contours, -1, (255, 255, 255), 1, cv2.LINE_AA) # border
    cvImg = cv2.drawContours(cvImg, contours, -1, color, 2)
    return cvImg

def draw_pose_info_text(cvImg, R, T, position=0, color=(0,255,0)):
    '''
    position = 0: left
    position = 1: right
    '''
    # get Euler angles (for display only)
    ai, aj, ak = transforms3d.euler.mat2euler(R, axes='szyx')
    #
    if position == 0:
        xPos = 20
    elif position == 1:
        xPos = cvImg.shape[1] - 300
    #
    info = ("Eas: %.2f, %.2f, %.2f" % (ak, aj, ai)) #Euler angles
    cvImg = cv2.putText(cvImg, info, (xPos, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 10, cv2.LINE_AA) # black background
    cvImg = cv2.putText(cvImg, info, (xPos, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA) 
    info = ("T: %.2f, %.2f, %.2f" % (T[0], T[1], T[2]))
    cvImg = cv2.putText(cvImg, info, (xPos, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 10, cv2.LINE_AA) 
    cvImg = cv2.putText(cvImg, info, (xPos, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA) 
    # info = ("Error (ADI): %.2f / %.2f = %.2f %%" % (err_3d, msDia, 100*err_3d/msDia))
    # cvImg = cv2.putText(cvImg, info, (xPos, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 10, cv2.LINE_AA) 
    # cvImg = cv2.putText(cvImg, info, (xPos, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA) 
    return cvImg

def draw_2d_keypoints(visImg, keypoints2d, radius):
    # keypoints2d: n*8*2
    colors = [(255, 0, 0), (0, 255, 0), (255, 255, 0), 
              (0, 0, 255), (255, 128, 0), (131, 255, 120), 
              (110, 255, 0), (255, 153, 255), (120, 130, 255), 
              (160, 0 ,255),(255, 127, 127), (255, 0, 150)]
    xy_predictions = keypoints2d.to('cpu').numpy()
    xs = xy_predictions[:,:,0]#n*8
    ys = xy_predictions[:,:,1]#n*8
    for ki in range(xs.shape[0]):
        for ji in range(xs.shape[1]):
            visImg = cv2.circle(visImg, (int(xs[ki][ji]), int(ys[ki][ji])), radius=radius, color=colors[ki], thickness=-1)

    return visImg

def draw_pose_metric_text(cvImg, error_3d, error_2d, estecm_valid, diameter, color=(0,255,0)):
    # centered
    xPos = int(cvImg.shape[1]/2) - 100
    #
    info = ("Error_3D: %.2f%%" % (100 * error_3d/diameter))
    cvImg = cv2.putText(cvImg, info, (xPos, 20+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 10, cv2.LINE_AA) # black background
    cvImg = cv2.putText(cvImg, info, (xPos, 20+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA) 
    # 
    info = ("Error_2D: %.2f pixels" % (error_2d))
    cvImg = cv2.putText(cvImg, info, (xPos, 40+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 10, cv2.LINE_AA) # black background
    cvImg = cv2.putText(cvImg, info, (xPos, 40+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA) 
    # 
    info = ("CSM: %d" % (estecm_valid))
    cvImg = cv2.putText(cvImg, info, (xPos, 60+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 10, cv2.LINE_AA) # black background
    cvImg = cv2.putText(cvImg, info, (xPos, 60+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA) 
    return cvImg

def visualize_pred(meta_info, pred, sym_types, meshes, bboxes_3d, diameters):
    cvImg = cv2.imread(meta_info['path'], cv2.IMREAD_UNCHANGED)
    # 
    cvRawImg = cvImg.copy()
    K = meta_info['K']
    text_drawing = True
    # text_drawing = False
    # 
    surfacePts = []
    for ms in meshes:
        pts = np.array(ms.vertices)
        tmp_index = np.random.choice(len(pts), 1000, replace=True)
        pts = pts[tmp_index]
        surfacePts.append(pts)
    #
    # draw ground truth poses
    for i in range(len(meta_info['class_ids'])):
        cls_id = meta_info['class_ids'][i]
        R = meta_info['rotations'][i]
        T = meta_info['translations'][i]
        # draw pose axis
        # cvImg = draw_bounding_box(cvImg, R, T, pt3d, tmpPoses.K, (0,255,0), 1)
        cvImg = draw_bounding_box(cvImg, R, T, np.array(bboxes_3d[cls_id]), K, (0,0,255))
        cvImg = draw_pose_axis(cvImg, R, T, bboxes_3d[cls_id], K)
        # cvImg = draw_pose_contour(cvImg, R, T, meshes[cls_id], K, (0,255,0))
        # cvImg = draw_z_axis(cvImg, R, T, meshes[cls_id], K, (0,255,0))
        # 
        # put some info on the image
        if text_drawing:
            cvImg = draw_pose_info_text(cvImg, R, T, position=0, color=(0,255,0))
    #
    # draw predicted poses
    for score, cls_id, R, T, xy2d in pred:
        isSym = (("cls_" + str(cls_id)) in sym_types)

        # assuming only one object (TODO)
        gtR = meta_info['rotations'][0]
        gtT = meta_info['translations'][0]
        err_3d, err_2d = compute_pose_diff(
            surfacePts[cls_id], K,
            gtR, gtT, R, T, isSym)
       
        try:
            # colors = plt.get_cmap('tab20', 1000).colors * 255
            # for pIdx in range(len(xs)):
            #     cvImg = cv2.circle(cvImg, (int(xs[pIdx]), int(ys[pIdx])), 3, colors[cls_id], -1)
            # draw pose axis
            # cvImg = draw_bounding_box(cvImg, R, T, pt3d, gtPoses.K, (0,255,0), 1)
            # cvImg = draw_bounding_box(cvImg, R, T, pt3d, gtPoses.K, (0,0,255), 1)
            cvImg = draw_pose_axis(cvImg, R, T, bboxes_3d[cls_id], K)
            cvImg = draw_2d_keypoints(cvImg, xy2d, 2)
            # cvImg = draw_pose_contour(cvImg, R, T, meshes[cls_id], K, (0, 0, 255))
            # cvImg = draw_z_axis(cvImg, R, T, meshes[cls_id], K, (0, 0, 255))
            
            # put some info on the image
            if text_drawing:
                cvImg = draw_pose_info_text(cvImg, R, T, position=1, color=(0, 0, 255))
        except:
            pass

    # cv2.imshow('predicted', cvImg)
    # cv2.waitKey(0)
    return cvImg

def remap_pose(srcK, srcR, srcT, pt3d, dstK, transM):
    ptCnt = len(pt3d)
    pts = np.matmul(transM, np.matmul(srcK, np.matmul(srcR, pt3d.transpose()) + srcT))
    xs = pts[0] / (pts[2] + 1e-8)
    ys = pts[1] / (pts[2] + 1e-8)
    xy2d = np.concatenate((xs.reshape(-1,1),ys.reshape(-1,1)), axis=1)
    # retval, rot, trans, inliers = cv2.solvePnPRansac(pt3d, xy2d, dstK, None, flags=cv2.SOLVEPNP_EPNP, reprojectionError=5.0)
    retval, rot, trans = cv2.solvePnP(pt3d.reshape(ptCnt,1,3), xy2d.reshape(ptCnt,1,2), dstK, None, flags=cv2.SOLVEPNP_EPNP)
    if retval:
        newR = cv2.Rodrigues(rot)[0]  # convert to rotation matrix
        newT = trans.reshape(-1, 1)

        newPts = np.matmul(dstK, np.matmul(newR, pt3d.transpose()) + newT)
        newXs = newPts[0] / (newPts[2] + 1e-8)
        newYs = newPts[1] / (newPts[2] + 1e-8)
        newXy2d = np.concatenate((newXs.reshape(-1,1),newYs.reshape(-1,1)), axis=1)
        diff_in_pix = np.linalg.norm(xy2d - newXy2d, axis=1).mean()

        
        return newR, newT, diff_in_pix
    else:
        print('Error in pose remapping!')
        return srcR, srcT, -1

def pose_symmetry_handling(R, sym_types):
    if len(sym_types) == 0:
        return R
    
    assert(len(sym_types) % 2 == 0)
    itemCnt = int(len(sym_types) / 2)

    for i in range(itemCnt):
        axis = sym_types[2*i]
        mod = sym_types[2*i + 1] * np.pi / 180
        if axis == 'X':
            ai, aj, ak = transforms3d.euler.mat2euler(R, axes='sxyz')
            ai = 0 if mod == 0 else np.fmod(ai, mod)
            R = transforms3d.euler.euler2mat(ai, aj, ak, axes='sxyz')
        elif axis == 'Y':
            ai, aj, ak = transforms3d.euler.mat2euler(R, axes='syzx')
            ai = 0 if mod == 0 else np.fmod(ai, mod)
            R = transforms3d.euler.euler2mat(ai, aj, ak, axes='syzx')
        elif axis == 'Z':
            ai, aj, ak = transforms3d.euler.mat2euler(R, axes='szyx')
            ai = 0 if mod == 0 else np.fmod(ai, mod)
            R = transforms3d.euler.euler2mat(ai, aj, ak, axes='szyx')
        else:
            print("symmetry axis should be 'X', 'Y' or 'Z'")
            assert(0)
    return R.astype(np.float32)

# define a function which returns an image as numpy array from figure
def get_img_from_matplotlib_fig(fig, dpi=300):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    return img

def visualize_accuracy_per_depth(
        accuracy_adi_per_class, 
        accuracy_rep_per_class, 
        accuracy_adi_per_depth, 
        accuracy_rep_per_depth, 
        depth_range):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    rep_keys = accuracy_rep_per_class[0].keys()
    adi_keys = accuracy_adi_per_class[0].keys()
    depth_bins = len(accuracy_rep_per_depth)
    assert(len(accuracy_adi_per_depth) == len(accuracy_rep_per_depth))
    ax1.set_title('Statistics of 2D error')
    ax1.set_xlabel('Depth')
    ax1.set_ylabel('Success Rate (%)')
    ax2.set_title('Statistics of 3D error')
    ax2.set_xlabel('Depth')
    # ax2.set_ylabel('Success Rate (%)')
    # ax2.yaxis.tick_right()
    for k in rep_keys:
        xs = np.arange(depth_range[0], depth_range[1], (depth_range[1]-depth_range[0])/depth_bins)
        ys = []
        for i in range(depth_bins):
            if k in accuracy_rep_per_depth[i]:
                ys.append(accuracy_rep_per_depth[i][k])
            else:
                ys.append(0)
        ys = np.array(ys)
        # 
        # xnew = np.linspace(depth_range[0], depth_range[1], 300) / 1000
        # ynew = UnivariateSpline(xs, ys, k=2, s=100)(xnew)
        # ax1.plot(xnew, ynew, label=k)
        ax1.plot(xs, ys, marker='o', label=k)
    for k in adi_keys:
        xs = np.arange(depth_range[0], depth_range[1], (depth_range[1]-depth_range[0])/depth_bins)
        ys = []
        for i in range(depth_bins):
            if k in accuracy_adi_per_depth[i]:
                ys.append(accuracy_adi_per_depth[i][k])
            else:
                ys.append(0)
        ys = np.array(ys)
        # 
        # xnew = np.linspace(depth_range[0], depth_range[1], 300) / 1000
        # ynew = UnivariateSpline(xs, ys, k=2, s=100)(xnew)
        # ax2.plot(xnew, ynew, label=k)
        ax2.plot(xs, ys, marker='o', label=k)
    ax1.legend(loc='lower right')
    ax2.legend(loc='upper right')
    ax1.grid()
    ax2.grid()
    matFig = get_img_from_matplotlib_fig(fig)
    # cv2.imshow("xx", matFig)
    # cv2.waitKey(0)
    return matFig
    
def print_accuracy_per_class(accuracy_adi_per_class,  accuracy_auc_per_class, accuracy_rep_per_class):
    assert(len(accuracy_adi_per_class) == len(accuracy_rep_per_class))
    classNum = len(accuracy_adi_per_class)

    firstMeet = True

    for clsIdx in range(classNum):
        if len(accuracy_adi_per_class[clsIdx]) == 0:
            continue

        if firstMeet:
            adi_keys = accuracy_adi_per_class[clsIdx].keys()
            auc_keys = accuracy_auc_per_class[clsIdx].keys()
            rep_keys = accuracy_rep_per_class[clsIdx].keys()

            titleLine = "\t"
            for k in adi_keys:
                titleLine += (k + ' ')
            for k in auc_keys:
                titleLine += (k + ' ')
            for k in rep_keys:
                titleLine += (k + ' ')
            print(titleLine)

            firstMeet = False

        line_per_class = ("cls_%02d" % clsIdx)
        for k in adi_keys:
            line_per_class += ('\t%.2f' % accuracy_adi_per_class[clsIdx][k])
        for k in auc_keys:
            line_per_class += ('\t%.2f' % accuracy_auc_per_class[clsIdx][k])
        for k in rep_keys:
            line_per_class += ('\t%.2f' % accuracy_rep_per_class[clsIdx][k])
        print(line_per_class)

def render_objects(meshes, ids, poses, K, w, h):
    '''
    '''
    assert(K[0][1] == 0 and K[1][0] == 0 and K[2][0] ==0 and K[2][1] == 0 and K[2][2] == 1)
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]
    objCnt = len(ids)
    assert(len(poses) == objCnt)
    
    # set background with 0 alpha, important for RGBA rendering
    scene = pyrender.Scene(bg_color=np.array([1.0, 1.0, 1.0, 0.0]), ambient_light=np.array([0.02, 0.02, 0.02, 1.0]))
    # pyrender.Viewer(scene, use_raymond_lighting=True)
    # camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    camera = pyrender.IntrinsicsCamera(fx=fx,fy=fy,cx=cx,cy=cy,znear=0.05,zfar=100000)
    camera_pose = np.eye(4)
    # reverse the direction of Y and Z, check: https://pyrender.readthedocs.io/en/latest/examples/cameras.html
    camera_pose[1][1] = -1
    camera_pose[2][2] = -1
    scene.add(camera, pose=camera_pose)
    light = pyrender.SpotLight(color=np.ones(3), intensity=4.0, innerConeAngle=np.pi/16.0, outerConeAngle=np.pi/6.0)
    # light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    # light = pyrender.PointLight(color=np.ones(3), intensity=4.0)
    scene.add(light, pose=camera_pose)
    for i in range(objCnt):
        clsId = int(ids[i])
        mesh = pyrender.Mesh.from_trimesh(meshes[clsId])

        H = np.zeros((4,4))
        H[0:3] = poses[i][0:3]
        H[3][3] = 1.0
        scene.add(mesh, pose=H)

    # pyrender.Viewer(scene, use_raymond_lighting=True)

    r = pyrender.OffscreenRenderer(w, h)
    # flags = pyrender.RenderFlags.OFFSCREEN | pyrender.RenderFlags.DEPTH_ONLY
    # flags = pyrender.RenderFlags.OFFSCREEN
    flags = pyrender.RenderFlags.OFFSCREEN | pyrender.RenderFlags.RGBA
    color, depth = r.render(scene, flags=flags)
    # color, depth = r.render(scene)
    # color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR) # RGB to BGR (for OpenCV)
    color = cv2.cvtColor(color, cv2.COLOR_RGBA2BGRA) # RGBA to BGRA (for OpenCV)
    # # 
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.axis('off')
    # plt.imshow(color)
    # plt.subplot(1,2,2)
    # plt.axis('off')
    # plt.imshow(depth, cmap=plt.cm.gray_r)
    # # plt.imshow(depth)
    # plt.show()
    # # 
    # r.delete()
    # color = None
    # 
    return color, depth

def compute_pose_diff(mesh3ds, K, gtR, gtT, predR, predT, isSym=False):
    ptCnt = len(mesh3ds)
    if ptCnt > 1000:
        tmp_index = np.random.choice(len(mesh3ds), 1000, replace=True)
        mesh3ds = mesh3ds[tmp_index]
        ptCnt = 1000
    #
    pred_3d1 = (np.matmul(gtR, mesh3ds.T) + gtT).T
    pred_3d2 = (np.matmul(predR, mesh3ds.T) + predT).T

    # find the closest point for symmetric objects
    if isSym:
        ext_3d1 = pred_3d1.repeat(ptCnt, axis=0)
        ext_3d2 = pred_3d2.reshape(1, -1).repeat(ptCnt, axis=0).reshape(-1, 3)
        min_idx2 = np.argmin(np.linalg.norm(ext_3d1-ext_3d2, axis=1).reshape(ptCnt, -1), axis=1)
        pred_3d2 = ext_3d2[min_idx2]

    p = np.matmul(K, pred_3d1.T)
    p[0] = p[0] / (p[2] + 1e-8)
    p[1] = p[1] / (p[2] + 1e-8)
    pred_2d1 = p[:2].T

    p = np.matmul(K, pred_3d2.T)
    p[0] = p[0] / (p[2] + 1e-8)
    p[1] = p[1] / (p[2] + 1e-8)
    pred_2d2 = p[:2].T

    error_3d = np.linalg.norm(pred_3d1 - pred_3d2, axis=1).mean()
    error_2d = np.linalg.norm(pred_2d1 - pred_2d2, axis=1).mean()

    return error_3d, error_2d

def compute_pose_diff_speed(gtR, gtT, predR, predT):
    q1 = rotation2quaternion(gtR)
    q2 = rotation2quaternion(predR)
    err_r = 2 * np.arccos(min(1, abs(q1.dot(q2))))
    err_t = np.linalg.norm(gtT-predT) / np.linalg.norm(gtT)
    return err_r, err_t

def evalute_auc_metric(error_3ds, max_err):
    error_3ds = np.array(error_3ds)
    sampleCnt = len(error_3ds)
    if sampleCnt == 0:
        return 0
    binCnt = 1000
    total_auc = 0.0
    for i in range(binCnt):
        validCnt = (error_3ds <= ((i+1) * (max_err/binCnt))).sum()
        binContrib = ((validCnt / sampleCnt) * (1 / binCnt))
        total_auc += binContrib
    return total_auc


def Procrustes_by_SVD(X, Y, W=None):
    #
    # input: each line of X or Y is a point
    #
    # will return R, T, scale
    # that's find these 3 best matrix to minimize W*||R*(X.t)+T - scale*Y.t||,
    # in which R is a rotation matrix
    #

    N = len(X)  # number of points
    assert (len(Y) == N)
    if W is None:
        W = torch.ones(N).type_as(X) / N
    else:
        W = W / W.sum()  # normalize weights
    W = W.view(-1, 1)

    # 1st step: translated first, so that their centroid coincides with the origin of the coordinate system.
    # This is done by subtracting from the point coordinates the coordinates of the respective centroid.
    ox = (X * W).sum(dim=0)  # mean X
    cx = X - ox  # center X
    # nx_factor = cx.norm()  # to normalize
    # nx = cx / nx_factor
    # print(nx)

    oy = (Y * W).sum(dim=0)  # mean Y
    cy = Y - oy  # center Y
    # ny_factor = cy.norm()  # to normalize
    # ny = cy / ny_factor
    # print(ny)

    # 2nd step: calculating a cross-covariance matrix A. or In matrix notation:
    # A = nx.t().mm(ny)
    A = cx.t().mm(W * cy)
    # print(A)

    # 3rd step
    U, S, V = torch.svd(A)
    # U,S,V = np.linalg.svd(A) # !!! V is already transposed
    # print(U)
    # print(S)
    # print(V)

    tmpR = V.mm(U.t())

    # compute determinant
    # det = tmpR[0][0] * tmpR[1][1] * tmpR[2][2] + tmpR[0][1] * tmpR[1][2] * tmpR[2][0] + tmpR[0][2] * tmpR[1][0] * \
    #       tmpR[2][1] \
    #       - tmpR[0][2] * tmpR[1][1] * tmpR[2][0] - tmpR[0][0] * tmpR[1][2] * tmpR[2][1] - tmpR[0][1] * tmpR[1][0] * \
    #       tmpR[2][2]
    det = torch.det(tmpR)
    # print(det)

    R = V.mm(torch.diag(torch.FloatTensor([1, 1, det]).type_as(A))).mm(U.t())

    scale = S.sum() / (W.mean() * (cy * cy).sum())
    T = R.mm(-ox.view(-1, 1)) + scale * oy.view(-1, 1)

    # err = (R.mm(X.t()) + T - scale * (Y.t())).pow(2).sum()
    # print(err)

    return R, T, scale

def solve_PnP_LHM(intrinsic, p3d, p2d, wts=None):
    # debug
    if False:
        # if True:
        p3d = torch.FloatTensor(
            [-1.2963, -0.3214, 0.8015, 0.2941, 0.5221, -1.0914, 0.9497, -1.1567, 1.6878, -0.3894, -0.0270, -0.1134,
             0.4587,
             -0.6117, 0.2934]).view(3, -1).t().double()
        px = torch.FloatTensor([-0.0774, 0.0043, 0.1498, 0.1484, 0.1268]).double()
        py = torch.FloatTensor([-0.0172, 0.3015, 0.1536, 0.4440, 0.1995]).double()
        wts = torch.ones(len(px)).double()
        intrinsic = torch.eye(3).double()

    p3d = p3d.view(-1, 3)
    p2d = p2d.view(-1, 2)
    N = len(p3d)
    if wts is None:
        wts = torch.ones(N).type_as(p3d) / N
    else:
        wts = wts / wts.sum()  # normalize weights

    # construct normalized 2d
    B = torch.inverse(intrinsic).mm(torch.cat((p2d.t(), torch.ones((1, len(p2d))).type_as(p2d)), dim=0)).t()

    A = p3d
    I = torch.eye(3).type_as(p3d)

    # compute projection matrices
    P = torch.bmm(B.view(-1, 3, 1), B.view(-1, 1, 3)) / torch.bmm(B.view(-1, 1, 3), B.view(-1, 3, 1))

    # compute the constant matrix factor required to compute t
    C = torch.inverse(I - (wts.view(-1, 1, 1) * P).sum(dim=0))

    # get init R
    Biter = B
    err_old = 1
    obj_err = 0
    TOL = 1e-5
    iter = 1
    while abs((err_old - obj_err) / err_old) > TOL:
        # for i in range(5):
        err_old = obj_err
        Riter, _, _ = Procrustes_by_SVD(A, torch.bmm(P, Biter.view(-1, 3, 1)).view(-1, 3), wts)
        # Riter, _, _, _ = Procrustes_by_quaternion(P, torch.bmm(V, Qiter.view(-1, 3, 1)).view(-1, 3)) #TODO, problem in back-propagation
        T = C.mm(torch.bmm(P - I, wts.view(-1, 1, 1) * Riter.mm(A.t()).t().view(-1, 3, 1)).sum(dim=0))
        Biter = (Riter.mm(A.t()) + T).t()

        # computer object error
        obj_err = torch.bmm(I - P, wts.view(-1, 1, 1) * Biter.view(-1, 3, 1)).view(-1, 3)
        obj_err = torch.sqrt((obj_err * obj_err).sum())

        # print(obj_err)

        iter += 1
        if iter > 20:
            break

    # computer image error
    Brep = torch.cat(((Biter[:, 0] / Biter[:, 2]).view(-1, 1),
                      (Biter[:, 1] / Biter[:, 2]).view(-1, 1),
                      torch.ones(N, 1).type_as(B)), dim=1)
    img_err = wts.view(-1, 1) * (Brep - B)
    img_err = torch.sqrt((img_err * img_err).sum())

    return Riter, T, [obj_err, img_err]


# https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
class JsonNumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# https://stackoverflow.com/questions/13249415/how-to-implement-custom-indentation-when-pretty-printing-with-the-json-module
def json_dumps_numpy(data, indent=2, depth=2):
    assert depth > 0
    space = ' '*indent
    s = json.dumps(data, indent=indent, cls=JsonNumpyEncoder)
    lines = s.splitlines()
    N = len(lines)
    # determine which lines to be shortened
    is_over_depth_line = lambda i: i in range(N) and lines[i].startswith(space*(depth+1))
    is_open_bracket_line = lambda i: not is_over_depth_line(i) and is_over_depth_line(i+1)
    is_close_bracket_line = lambda i: not is_over_depth_line(i) and is_over_depth_line(i-1)
    # 
    def shorten_line(line_index):
        if not is_open_bracket_line(line_index):
            return lines[line_index]
        # shorten over-depth lines
        start = line_index
        end = start
        while not is_close_bracket_line(end):
            end += 1
        has_trailing_comma = lines[end][-1] == ','
        _lines = [lines[start][-1], *lines[start+1:end], lines[end].replace(',','')]
        d = json.dumps(json.loads(' '.join(_lines)))
        return lines[line_index][:-1] + d + (',' if has_trailing_comma else '')
    # 
    s = '\n'.join([
        shorten_line(i)
        for i in range(N) if not is_over_depth_line(i) and not is_close_bracket_line(i)
    ])
    #
    return s



def get_dataset_dict(image_files, objID_2_clsID, mem_cache=None, cache_dir='.cache', name='lmo', data='None'):
    os.makedirs(f"{cache_dir}", exist_ok=True)
    cache_path = osp.join(cache_dir, "dataset_dicts_{}_{}.pkl".format(name, data))
    if osp.exists(cache_path):
        file = open(cache_path, 'rb') 
        dataset_dicts = pickle.load(file)
        file.close()
        return dataset_dicts

    dataset_dicts = []
    for img_path in tqdm(image_files):
        img_path = img_path.strip()
        # 
        gt_dir, tmp, imgName = img_path.rsplit('/', 2)
        assert(tmp == 'rgb')
        imgBaseName, _ = os.path.splitext(imgName)
        # 
        camera_file = gt_dir + '/scene_camera.json'
        gt_file = gt_dir + "/scene_gt.json"
        # gt_info_file = gt_dir + "/scene_gt_info.json"
        gt_mask_visib = gt_dir + "/mask_visib/"

        gt_json = load_json_cached(gt_file, mem_cache)
        cam_json = load_json_cached(camera_file, mem_cache)
        # gt_info_json = json.load(open(gt_info_file))

        im_id_str = str(int(imgBaseName))
        if im_id_str in cam_json:
            annot_camera = cam_json[im_id_str]
        else:
            annot_camera = cam_json[imgBaseName]
        # 
        if im_id_str in gt_json:
            annot_poses = gt_json[im_id_str]
        else:
            annot_poses = gt_json[imgBaseName]
    
        objCnt = len(annot_poses)
        K = np.array(annot_camera['cam_K']).reshape(3,3)

        # merged_mask = None
        # instance_idx = 1
       
        for i in range(objCnt):

            obj_id = str(annot_poses[i]['obj_id'])
            if not obj_id in objID_2_clsID:
                continue
            cls_id = objID_2_clsID[obj_id]
       
            mask_vis_file = gt_mask_visib + ("%s_%06d.png" %(imgBaseName, i))
            # mask_vis = load_image_cached(mask_vis_file, mem_cache)
            # height = mask_vis.shape[0]
            # width = mask_vis.shape[1]
            # if merged_mask is None:
            #     merged_mask = np.zeros((height, width), np.uint8) # segmenation masks
        
            R = np.array(annot_poses[i]['cam_R_m2c']).reshape(3,3)
            T = np.array(annot_poses[i]['cam_t_m2c']).reshape(3,1)
            
            # compose segmentation labels
            # merged_mask[mask_vis==255] = instance_idx
            # instance_idx += 1

            record = {
                "img_path": img_path,
                "K": K,
                "nid": i,
                "obj_id": obj_id, 
                "cls_id": [cls_id],
                "rotation": [R],
                "translations": [T],
                "mask_file": mask_vis_file

            }

            dataset_dicts.append(record)
    file= open(cache_path, 'wb')
    pickle.dump(dataset_dicts, file)
    file.close()
    
    return dataset_dicts

def load_mask(mask_file, mem_cache):
    instance_idx = 1
    mask_vis = load_image_cached(mask_file, mem_cache)
    height = mask_vis.shape[0]
    width = mask_vis.shape[1]
            
    merged_mask = np.zeros((height, width), np.uint8) # segmenation masks
    # compose segmentation labels
    merged_mask[mask_vis==255] = instance_idx

    return merged_mask
