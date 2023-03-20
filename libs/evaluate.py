import json

import cv2
import transforms3d

import copy
import numpy as np
import matplotlib.pyplot as plt
import random

from libs.utils import (
    remap_pose,
    get_single_bop_annotation,
    load_bop_meshes,
    load_bbox_3d,
    pose_symmetry_handling,
    evalute_auc_metric,
    compute_pose_diff,
    compute_pose_diff_speed
)

from libs.poses import PoseAnnot

def evaluate_pose_predictions(predictions, class_number, meshes, mesh_diameters, symmetry_types):
    INF = 100000000
    classNum = class_number - 1 # get rid of the background class

    thresholds_adi = [0.05, 0.10, 0.20, 0.50]
    thresholds_rep = [2, 5, 10, 20]
        
    accuracy_adi_per_class = []
    accuracy_auc_per_class = []
    accuracy_rep_per_class = []
    # 
    depth_bins = 3
    accuracy_adi_per_depth = []
    accuracy_rep_per_depth = []

    # get depth range from annotations, and divide it to serval bins
    depth_min = INF
    depth_max = 0
    for filename, item in predictions.items():
        gtTs = np.array(item['meta']['translations'])
        for T in gtTs:
            depth = float(T.reshape(-1)[2])
            depth_min = min(depth_min, depth)
            depth_max = max(depth_max, depth)
    depth_max += 1e-5 # add some margin for safe depth index computation
    depth_bin_width = (depth_max - depth_min) / depth_bins

    errors_adi_per_depth = list([] for i in range(0, depth_bins))
    errors_rep_per_depth = list([] for i in range(0, depth_bins))
    for clsid in range(classNum):
        isSym = (("cls_" + str(clsid)) in symmetry_types)
        errors_adi_all = [] # 3D errors, %
        errors_abs3d_all = [] # 3D erros, absolute
        errors_rep_all = [] # 2D errors
        errors_speed_all = [] # in speed metric
        depth_all = [] # depth for each sample
        object_cx_all = []
        object_cy_all = []
        # 
        for filename, item in predictions.items():
            K = np.array(item['meta']['K'])
            pred = item['pred']
            gtIDs = item['meta']['class_ids']
            gtRs = np.array(item['meta']['rotations'])
            gtTs = np.array(item['meta']['translations'])
            
            # filter by class id
            pred = [p for p in pred if p[1] == clsid]
            gtIdx = [gi for gi in range(len(gtIDs)) if gtIDs[gi] == clsid]
            if len(gtIdx) == 0:
                continue

            # find predictions with best confidences
            assert(len(gtIdx) == 1) # only one object for one class now

            # get the depth bin of the object
            gi = gtIdx[0] # only pick up the first one
            depth = float(gtTs[gi].reshape(-1)[2])
            depth_idx = int((depth - depth_min) / depth_bin_width)
            depth_all.append(depth)
            # 
            if len(pred) > 0:
                # find the best confident one
                bestIdx = 0
                R1 = gtRs[gi]
                T1 = gtTs[gi]
                R2 = np.array(pred[bestIdx][2])
                T2 = np.array(pred[bestIdx][3])
                err_3d, err_2d = compute_pose_diff(meshes[clsid].vertices, K, R1, T1, R2, T2, isSym=isSym)
                # 
                err_r, err_t = compute_pose_diff_speed(R1, T1, R2, T2)
                errors_speed_all.append(err_r + err_t)
                #
                # 
                # get the reprojected center
                tmp_pt = np.matmul(K, T1)
                object_cx = tmp_pt[0] / tmp_pt[2]
                object_cy = tmp_pt[1] / tmp_pt[2]
                object_cx_all.append(float(object_cx))
                object_cy_all.append(float(object_cy))
                # 
                errors_adi_all.append(err_3d / mesh_diameters[clsid])
                errors_abs3d_all.append(err_3d)
                errors_rep_all.append(err_2d)
                errors_adi_per_depth[depth_idx].append(err_3d / mesh_diameters[clsid])
                errors_rep_per_depth[depth_idx].append(err_2d)
            else:
                object_cx_all.append(-1)
                object_cy_all.append(-1)
                errors_adi_all.append(1.0)
                errors_abs3d_all.append(1e10)
                errors_rep_all.append(50)
                errors_speed_all.append(100)
                errors_adi_per_depth[depth_idx].append(1.0)
                errors_rep_per_depth[depth_idx].append(50)
        # 
        auc = evalute_auc_metric(errors_abs3d_all, max_err=100)
        # 
        err_vs_pos = np.stack((np.array(object_cx_all), np.array(object_cy_all), np.array(errors_adi_all))).transpose()
        dis_to_center = np.sqrt((np.array(object_cx_all) - 512)*(np.array(object_cx_all) - 512) + (np.array(object_cy_all) - 512)*(np.array(object_cy_all) - 512))
        erro_vs_center_dis = np.stack((dis_to_center, np.array(errors_adi_all))).transpose()
        # np.savetxt('out.txt', erro_vs_center_dis, fmt='%.3f')
        estec_m = 0
      
        assert(len(errors_adi_all) == len(errors_rep_all))
        counts_all = len(errors_adi_all)
        if counts_all > 0:
            accuracy = {}
            for th in thresholds_adi:
                validCnt = (np.array(errors_adi_all) < th).sum()
                key = 'ADI' + ("%.2fd" % th).lstrip('0')
                accuracy[key] = (validCnt / counts_all) * 100
            accuracy_adi_per_class.append(accuracy)
            # 
            accuracy = {}
            accuracy['AUC    '] = auc * 100
            accuracy_auc_per_class.append(accuracy)
            # 
            accuracy = {}
            for th in thresholds_rep:
                validCnt = (np.array(errors_rep_all) < th).sum()
                accuracy[('REP%02dpx'%th)] = (validCnt / counts_all) * 100
            accuracy_rep_per_class.append(accuracy)
        else:
            accuracy_adi_per_class.append({})
            accuracy_auc_per_class.append({})
            accuracy_rep_per_class.append({})
    # 
    # compute accuracy for every depth bin
    for i in range(depth_bins):
        assert(len(errors_adi_per_depth[i]) == len(errors_rep_per_depth[i]))
        counts_all = len(errors_adi_per_depth[i])
        if counts_all > 0:
            accuracy = {}
            for th in thresholds_adi:
                validCnt = (np.array(errors_adi_per_depth[i]) < th).sum()
                key = 'ADI' + ("%.2fd" % th).lstrip('0')
                accuracy[key] = (validCnt / counts_all) * 100
            accuracy_adi_per_depth.append(accuracy)
            accuracy = {}
            for th in thresholds_rep:
                validCnt = (np.array(errors_rep_per_depth[i]) < th).sum()
                accuracy[('REP%02dpx'%th)] = (validCnt / counts_all) * 100
            accuracy_rep_per_depth.append(accuracy)
        else:
            accuracy_adi_per_depth.append({})
            accuracy_rep_per_depth.append({})
    # 
    return accuracy_adi_per_class, accuracy_auc_per_class, accuracy_rep_per_class, accuracy_adi_per_depth, accuracy_rep_per_depth, [depth_min, depth_max]

def remap_predictions(internal_K, internal_width, internal_height, keypoints_3d, meta, preds):
    new_preds = []
    internal_K = np.array(internal_K).reshape(3,3)
    # 
    K = np.array(meta['K'])
    width = meta['width']
    height = meta['height']
    class_ids = meta['class_ids']
    rotations = np.array(meta['rotations'])
    translations = np.array(meta['translations'])
    # 
    for idx in range(len(preds)):
        score, clsid, R, T, xy2d = preds[idx]
        pt3d = np.array(keypoints_3d[clsid])
        newR, newT, remap_err = remap_pose(
            internal_K, np.array(R), np.array(T), 
            np.array(pt3d), K, 
            np.matmul(K, np.linalg.inv(internal_K))
            )
        new_preds.append([score, clsid, newR, newT, xy2d])
    # 
    return new_preds

if __name__ == "__main__":
    image_path = "/data/BOP/ycbv/train_real/000040/rgb/000001.png"
    bbox_json = "/data/BOP/ycbv/ycb_bbox.json"
    meshpath = '/data/BOP/ycbv/models_eval/'
   
    meshes, objID_2_clsID = load_bop_meshes(meshpath)
    K, merged_mask, class_ids, rotations, translations = \
        get_single_bop_annotation(image_path, objID_2_clsID)
    cvImg = cv2.imread(image_path)
    height, width, _ = cvImg.shape
    # 
    with open(bbox_json, 'r') as f:
        bbox_3d = json.load(f)
    # 
    poses = PoseAnnot(bbox_3d, K, merged_mask, class_ids, rotations, translations, width, height)
    cvImg = poses.visualize(cvImg)
    cv2.imshow("img", cvImg)
    cv2.waitKey(0)

    # 
    # idx = 4 # bowl
    idx = 0
    clsid = class_ids[idx]
    R1 = rotations[idx]
    T1 = translations[idx]
    R2, T2 = pose_symmetry_handling(R1, T1, ['Y',0])
    mesh3ds = np.array(meshes[clsid].vertices)
    error_3d, error_2d = compute_pose_diff(mesh3ds, K, R1, T1, R2, T2, isSym=False)
    error_3d, error_2d = compute_pose_diff(mesh3ds, K, R1, T1, R2, T2, isSym=True)
    pass
