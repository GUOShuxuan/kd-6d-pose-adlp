def kd_loss_2d(pred_xy, target_xy, pred_cls, target_cls, w, h, level, kd_loss, dim, pos_per_img=None, pos_per_img_t=None, normalize=True):
    """
    pred_xy: npos*8,2
    target_xy: npos*8, 2
    """
  
    if dim==2:
        if normalize:
            pred_xy[:, 0] = pred_xy[:, 0] / w 
            pred_xy[:, 1] = pred_xy[:, 1] / h
            target_xy[:, 0] = target_xy[:, 0] / w
            target_xy[:, 1] = target_xy[:, 1] / h

    losses = [] 

    nimg = len(pos_per_img)
    pred_xy = pred_xy.view(-1, 8, dim)
    target_xy = target_xy.view(-1, 8, dim)
    start = 0
    start_t = 0

    for img_id in range(nimg):
        end = start + pos_per_img[img_id]
        end_t = start_t + pos_per_img_t[img_id]
        if (start == end) or (start_t == end_t) :
            start = end
            start_t = end_t
            continue  
        pred_xy_t = target_xy[start_t:end_t, ...]
        pred_xy_s = pred_xy[start:end, ...]

        if target_cls is not None:
            pred_cls_t = target_cls[start_t:end_t, ...] # 137 *8 ==> 7*8
            pred_cls_s = pred_cls[start:end, ...]

        start = end
        start_t = end_t
       
        if level == 'point':
       
            pred_xy_t = pred_xy_t.transpose(0,1).contiguous() # 8 * npoints * 2
            pred_xy_s = pred_xy_s.transpose(0,1).contiguous()
            if target_cls is not None: 
              
                pred_cls_t = pred_cls_t.transpose(0,1).contiguous()
                pred_cls_s = pred_cls_s.transpose(0,1).contiguous()
                loss = kd_loss(pred_cls_s, pred_xy_s, pred_cls_t, pred_xy_t).sum()  
            else:
                loss = kd_loss(pred_xy_s, pred_xy_t).sum() 
        losses.append(loss)
    return losses

