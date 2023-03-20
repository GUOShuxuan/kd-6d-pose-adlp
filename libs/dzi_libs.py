## refer to SO-Pose: https://github.com/shangbuhuan13/SO-Pose
import numpy as np
import cv2
import torch
from libs.poses import PoseAnnot

_DZI_TYPE='uniform'
_DZI_PAD_SCALE = 1.5 
_DZI_SCALE_RATIO=0.25
_DZI_SHIFT_RATIO=0.25

_INPUT_RES = 256

def aug_bbox_DZI(bbox_xyxy, im_H, im_W):
    """Used for DZI, the augmented box is a square (maybe enlarged)
    Args:
        bbox_xyxy (np.ndarray):
    Returns:
            center, scale
    """
    x1, y1, x2, y2 = bbox_xyxy.detach().numpy()
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bh = y2 - y1
    bw = x2 - x1

    if _DZI_TYPE.lower() == "uniform":
        scale_ratio = 1 + _DZI_SCALE_RATIO * (2 * np.random.random_sample() - 1)  # [1-0.25, 1+0.25]
        shift_ratio = _DZI_SHIFT_RATIO * (2 * np.random.random_sample(2) - 1)  # [-0.25, 0.25]
        bbox_center = np.array([cx + bw * shift_ratio[0], cy + bh * shift_ratio[1]])  # (h/2, w/2)
        scale = max(y2 - y1, x2 - x1) * scale_ratio * _DZI_PAD_SCALE
    elif _DZI_TYPE.lower() == "roi10d":
        # shift (x1,y1), (x2,y2) by 15% in each direction
        _a = -0.15
        _b = 0.15
        x1 += bw * (np.random.rand() * (_b - _a) + _a)
        x2 += bw * (np.random.rand() * (_b - _a) + _a)
        y1 += bh * (np.random.rand() * (_b - _a) + _a)
        y2 += bh * (np.random.rand() * (_b - _a) + _a)
        x1 = min(max(x1, 0), im_W)
        x2 = min(max(x1, 0), im_W)
        y1 = min(max(y1, 0), im_H)
        y2 = min(max(y2, 0), im_H)
        bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])
        scale = max(y2 - y1, x2 - x1) * _DZI_PAD_SCALE
    elif _DZI_TYPE.lower() == "truncnorm":
        raise NotImplementedError("DZI truncnorm not implemented yet.")
    else:
        bbox_center = np.array([cx, cy])  # (w/2, h/2)
        scale = max(y2 - y1, x2 - x1)
    scale = min(scale, max(im_H, im_W)) * 1.0
    return bbox_center, scale


def dzi_train(img, target):

    im_H = img.size(1)
    im_W = img.size(2)

    bbox_xyxy = target.to_object_boxlist().bbox[0]
    bbox_center, scale = aug_bbox_DZI(bbox_xyxy, im_H, im_W)

    input_res = _INPUT_RES
    image = img.detach().numpy().transpose(1,2,0)#.numpy()

    # get roi img
    roi_img, trans_img = crop_resize_by_warp_affine(
            image, bbox_center, scale, input_res, interpolation=cv2.INTER_LINEAR
        ) # 3, 256, 256

    # transform gt_bbox, mask 
    mask = target.mask.numpy() #480, 640
    roi_mask_visib, trans_mask = crop_resize_by_warp_affine(
            mask[:, :, None], bbox_center, scale, input_res, interpolation=cv2.INTER_NEAREST
        )


    target.mask = torch.as_tensor(roi_mask_visib.astype("float32")).contiguous()

    
    scale_r = input_res / scale
    scale_r = torch.as_tensor(scale_r).contiguous() 
    trans_img = torch.as_tensor((trans_img).astype("float32")).contiguous()

    target.add_bbox_scale(scale_r)
    target.add_bbox_trans(trans_img)

    

    target.width = input_res
    target.height = input_res
   
    img = torch.as_tensor(roi_img.transpose(2, 0, 1).astype("float32")).contiguous()
    
    return img, target

def dzi_test(img, target, det=None):
    # adjust target.bbox; mask
    im_H = img.size(1)
    im_W = img.size(2)

    bbox_xyxy = target.to_object_boxlist().bbox[0]
    x1, y1, x2, y2 = bbox_xyxy.detach().numpy()
    bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])
    bw = max(x2 - x1, 1)
    bh = max(y2 - y1, 1)
    scale = max(bh, bw) * _DZI_PAD_SCALE 
    scale = min(scale, max(im_H, im_W)) * 1.0
    
    input_res = _INPUT_RES
    image = img.detach().numpy().transpose(1,2,0)#.numpy()
    # get roi img
    roi_img, trans_img = crop_resize_by_warp_affine(
            image, bbox_center, scale, input_res, interpolation=cv2.INTER_LINEAR
        )
    roi_img = roi_img.transpose(2, 0, 1) # 3, 256, 256

    # transform gt_bbox, mask 
    mask = target.mask.numpy() #480, 640
    roi_mask_visib, trans_mask = crop_resize_by_warp_affine(
            mask[:, :, None], bbox_center, scale, input_res, interpolation=cv2.INTER_NEAREST
        )

    target.mask = torch.as_tensor(roi_mask_visib.astype("float32")).contiguous()
    

    scale_r = input_res / scale
    scale_r = torch.as_tensor(scale_r.astype("float32")).contiguous()
 
    trans_img = torch.as_tensor((trans_img).astype("float32")).contiguous()

    target.add_bbox_scale(scale_r)
    target.add_bbox_trans(trans_img)


    target.width = input_res
    target.height = input_res
    img = torch.as_tensor(roi_img.astype("float32")).contiguous()
    
    return img, target

def crop_resize_by_warp_affine(img, center, scale, output_size, rot=0, interpolation=cv2.INTER_LINEAR):
    """
    output_size: int or (w, h)
    NOTE: if img is (h,w,1), the output will be (h,w)
    """
    if isinstance(scale, (int, float)):
        scale = (scale, scale)
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img, trans, (int(output_size[0]), int(output_size[1])), flags=interpolation)

    return dst_img, trans

def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=False):
    """
    adapted from CenterNet: https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
    center: ndarray: (cx, cy)
    scale: (w, h)
    rot: angle in deg
    output_size: int or (w, h)
    """
    if isinstance(center, (tuple, list)):
        center = np.array(center, dtype=np.float32)

    if isinstance(scale, (int, float)):
        scale = np.array([scale, scale], dtype=np.float32)

    if isinstance(output_size, (int, float)):
        output_size = (output_size, output_size)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def dzi_test_mobj(img, targets, det=None):
    # adjust target.bbox; mask
    # collect all roi_imgs and target for multiple objects
    nobj = targets.__len__()
    roi_targets = []
    roi_imgs = []
    for i in range(nobj):
        mask = (targets.mask == (i+1)).float()
        target =  PoseAnnot(
            torch.FloatTensor(targets.keypoints_3d),
            torch.FloatTensor(targets.K),
            torch.FloatTensor(mask),
            torch.LongTensor(targets.class_ids[i].view(-1)),
            torch.FloatTensor(np.array(targets.rotations[i].unsqueeze(0))),
            torch.FloatTensor(np.array(targets.translations[i].unsqueeze(0))),
            targets.width, targets.height,
            )
        roi_img, roi_target = dzi_test(img, target)
        roi_imgs.append(roi_img)
        roi_targets.append(roi_target)
    return roi_imgs, roi_targets

_GREEN = (18, 127, 15)
def vis_image_mask_bbox_cv2(
    img, mask, bbox=None, labels=None, font_scale=0.5, text_color="green", font_thickness=2, box_thickness=1
):
    """
    bboxes: xyxy
    """
    # import pycocotools.mask as cocomask
    # text_color = color_val(text_color)
    img_show = img.copy()
    
    color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
    mask = mask.astype(np.bool)
    img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
    
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img_show, (int(x1), int(y1)), (int(x2), int(y2)), _GREEN, thickness=box_thickness)
        
    return img_show


