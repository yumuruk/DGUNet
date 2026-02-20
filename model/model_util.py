import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
from collections import defaultdict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_channel_statistics(batch):
    batch_size = batch.shape[0]
    
    sorted_channel_means = []      
    sorted_channel_indices = []  

    large_channel_idx = []
    medium_channel_idx = []
    small_channel_idx = []

    large_channel_val = []
    medium_channel_val = []
    small_channel_val = []

    for batch_index in range(batch_size):
        image = batch[batch_index, :, :, :]  
        mean = torch.mean(image, (2, 1))   
        
        mean_sorted, channel_sorted_indices = torch.sort(mean) 
        sorted_channel_means.append(mean_sorted)
        sorted_channel_indices.append(channel_sorted_indices)

        large_channel_idx.append(channel_sorted_indices[2])
        medium_channel_idx.append(channel_sorted_indices[1])
        small_channel_idx.append(channel_sorted_indices[0])

        large_channel_val.append(image[channel_sorted_indices[2]].unsqueeze(0))
        medium_channel_val.append(image[channel_sorted_indices[1]].unsqueeze(0))
        small_channel_val.append(image[channel_sorted_indices[0]].unsqueeze(0))

    sorted_channel_means = torch.stack(sorted_channel_means)       
    sorted_channel_indices = torch.stack(sorted_channel_indices)      
    
    large_channel_idx = torch.stack(large_channel_idx)            
    medium_channel_idx = torch.stack(medium_channel_idx)
    small_channel_idx = torch.stack(small_channel_idx)

    large_channel_val = torch.stack(large_channel_val)       
    medium_channel_val = torch.stack(medium_channel_val)
    small_channel_val = torch.stack(small_channel_val)
    
    return sorted_channel_means, sorted_channel_indices, large_channel_val, medium_channel_val, small_channel_val, large_channel_idx, medium_channel_idx, small_channel_idx

def replace_channel_by_index(J, J_m, channel_idx):
    batch_size = J.shape[0]
    mapped_J = []
    for batch_index in range(batch_size):
        image = J[batch_index, :, :, :]
        image[channel_idx[batch_index]] = J_m[batch_index] # 
        mapped_J.append(image)
    mapped_J = torch.stack(mapped_J) 
    return mapped_J    
    
    
def get_dark_channel(x, patch_size):
    pad_size = (patch_size - 1) // 2
    H, W = x.size()[2], x.size()[3]
    x, _ = x.min(dim=1, keepdim=True)  
    x = nn.ReflectionPad2d(pad_size)(x)  
    x = nn.Unfold(patch_size)(x) 
    x = x.unsqueeze(1)  

    dark_map, index_map = x.min(dim=2, keepdim=False) 
    dark_map = dark_map.view(-1, 1, H, W)

    return dark_map, index_map

def softThresh(x, lamda):
    relu = nn.ReLU()
    return torch.sign(x).to(DEVICE) * relu(torch.abs(x).to(DEVICE) - lamda)


@torch.inference_mode()
def build_classwise_L_det(results, J_input: torch.Tensor, num_classes: int, conf_thres: float = 0.25, eps: float = 1e-6):
    device = J_input.device
    B, _, H, W = J_input.shape

    diag_dict = {}

    for b in range(B):
        r = results[b]

        if getattr(r, "boxes", None) is None or r.boxes is None:
            boxes = torch.empty((0, 4), device=device)
            confs = torch.empty((0,), device=device)
            clses = torch.empty((0,), device=device, dtype=torch.long)
        else:
            boxes = r.boxes.xyxy.to(device).float()
            confs = r.boxes.conf.to(device).float()
            clses = r.boxes.cls.to(device).long()

        if confs.numel() > 0:
            keep = confs >= conf_thres
            boxes = boxes[keep]
            clses = clses[keep]
        else:
            boxes = boxes[:0]
            clses = clses[:0]

        m_pred = torch.zeros((num_classes, H, W), dtype=torch.float32, device=device)

        if boxes.numel() > 0:
            x1 = boxes[:, 0].floor().clamp(0, W).long()
            y1 = boxes[:, 1].floor().clamp(0, H).long()
            x2 = boxes[:, 2].ceil().clamp(0, W).long()
            y2 = boxes[:, 3].ceil().clamp(0, H).long()

            valid = (x2 > x1) & (y2 > y1) & (clses >= 0) & (clses < num_classes)
            x1, y1, x2, y2, clses = x1[valid], y1[valid], x2[valid], y2[valid], clses[valid]

            for j in range(x1.numel()):
                c = int(clses[j])
                m_pred[c, int(y1[j]):int(y2[j]), int(x1[j]):int(x2[j])] = 1.0

        J_safe = J_input[b].clamp_min(eps)

        for k in range(num_classes):
            roi2d = m_pred[k]
            diag_vals = (roi2d.unsqueeze(0) / J_safe).reshape(-1).contiguous()
            diag_dict[(b, k)] = diag_vals

    return diag_dict

def xywh_label_to_xyxy_label(lb : torch.Tensor, H : int, W :int):
    lb = lb.float()
    xyxy_lb = torch.zeros_like(lb, dtype=torch.float32)
    xyxy_lb[:, 1] = ((lb[:, 1] - lb[:, 3] / 2) * W).floor().clamp(0.0, float(W))  # x1
    xyxy_lb[:, 2] = ((lb[:, 2] - lb[:, 4] / 2) * H).floor().clamp(0.0, float(H))  # y1
    xyxy_lb[:, 3] = ((lb[:, 1] + lb[:, 3] / 2) * W).ceil().clamp(0.0, float(W))   # x2
    xyxy_lb[:, 4] = ((lb[:, 2] + lb[:, 4] / 2) * H).ceil().clamp(0.0, float(H))   # y2

    xyxy_lb[:, 0] = lb[:, 0] 
    return xyxy_lb


@torch.inference_mode()
def build_classwise_M_gt(labels: torch.Tensor, J_input: torch.Tensor, num_classes: int):
    """Build class-wise GT masks aligned with L_det layout."""
    device = J_input.device
    B, _, H, W = J_input.shape

    M_gt = {}

    for b in range(B):
        m = torch.zeros((num_classes, H, W), dtype=torch.float32, device=device)

        sel = labels[:, 0] == float(b)
        lb = labels[sel, 1:6]  # (cls, cx, cy, w, h)

        if lb.numel() > 0:
            xyxy = xywh_label_to_xyxy_label(lb.to(device).float(), H, W)

            cls = xyxy[:, 0].long()
            x1  = xyxy[:, 1].long()
            y1  = xyxy[:, 2].long()
            x2  = xyxy[:, 3].long()
            y2  = xyxy[:, 4].long()

            valid = (x2 > x1) & (y2 > y1) & (cls >= 0) & (cls < num_classes)
            cls, x1, y1, x2, y2 = cls[valid], x1[valid], y1[valid], x2[valid], y2[valid]

            for j in range(x1.numel()):
                c = int(cls[j])
                m[c, int(y1[j]):int(y2[j]), int(x1[j]):int(x2[j])] = 1.0

        for k in range(num_classes):
            M_gt[(b, k)] = m[k].unsqueeze(0).expand(3, -1, -1).reshape(-1).contiguous()

    return M_gt



























# def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
#     """
#     PyTorch-only letterbox: resize + padding without OpenCV.
#     Args:
#         img: [C, H, W] or [B, C, H, W] float tensor in range [0, 1]
#     Returns:
#         img_lb: padded image
#         ratio: (r, r)
#         pad: (left, top)
#     """
#     single_input = False
#     if img.dim() == 3:
#         img = img.unsqueeze(0)
#         single_input = True

#     B, C, H, W = img.shape
#     r = min(new_shape[0] / H, new_shape[1] / W)
#     new_unpad = (int(round(W * r)), int(round(H * r)))  # (W', H')
    
#     # 1. resize
#     img_resized = F.interpolate(img, size=(new_unpad[1], new_unpad[0]), mode='bilinear', align_corners=False)

#     # 2. padding
#     dw = new_shape[1] - new_unpad[0]
#     dh = new_shape[0] - new_unpad[1]
#     left, right = dw // 2, dw - dw // 2
#     top, bottom = dh // 2, dh - dh // 2

#     pad = [left, right, top, bottom]  # F.pad expects [left, right, top, bottom]
#     img_padded = F.pad(img_resized, pad=[pad[0], pad[1], pad[2], pad[3]], mode='constant', value=color[0] / 255.0)

#     ratio = (r, r)
#     pad_vals = (left, top)

#     if single_input:
#         img_padded = img_padded.squeeze(0)

#     return img_padded, ratio, pad_vals



# def clip_coords(boxes, shape):
#     """
#     Clamps bounding xyxy coordinates to image shape (height, width).
#     Args:
#         boxes (Tensor): [N, 4] (xyxy)
#         shape (tuple): (height, width)
#     """
#     boxes[..., 0].clamp_(0, shape[1])  # x1
#     boxes[..., 1].clamp_(0, shape[0])  # y1
#     boxes[..., 2].clamp_(0, shape[1])  # x2
#     boxes[..., 3].clamp_(0, shape[0])  # y2

# def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
#     """
#     Rescale bounding box coords (xyxy) from letterboxed img1_shape to original img0_shape.
#     Args:
#         img1_shape (tuple): Shape of the network input image, e.g., (640, 640)
#         coords (Tensor): [N, 4] boxes in xyxy format, letterbox 기준 좌표
#         img0_shape (tuple): Shape of the target/original image, e.g., (H, W)
#         ratio_pad (tuple, optional): ((gain, gain), (padw, padh)). If None, 자동 계산
#     Returns:
#         coords (Tensor): [N, 4] boxes, 원본 해상도 기준으로 변환됨
#     """
#     if ratio_pad is None:
#         gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
#         pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # (padw, padh)
#     else:
#         gain = ratio_pad[0][0]
#         pad = ratio_pad[1]

#     # x1, x2에서 padw 빼고, y1, y2에서 padh 빼기 (box 전체 이동)
#     coords[..., [0, 2]] -= pad[0]
#     coords[..., [1, 3]] -= pad[1]
#     coords[..., :4] /= gain

#     clip_coords(coords, img0_shape)
#     return coords

# def draw_boxes(image, result, class_names=None, conf_thresh=0.3):
#     image_out = image.copy()
#     boxes = result.boxes.xyxy.cpu().numpy()
#     confs = result.boxes.conf.cpu().numpy()
#     classes = result.boxes.cls.cpu().numpy().astype(int)

#     for box, conf, cls_id in zip(boxes, confs, classes):
#         if conf < conf_thresh:
#             continue
#         x1, y1, x2, y2 = map(int, box)
#         label = f"{class_names[cls_id] if class_names else cls_id} {conf:.2f}"
#         cv2.rectangle(image_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(image_out, label, (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
#     return image_out



# def build_sparse_M_det_from_yolo_results(results, img_shape, conf_thresh=0.5):
#     """
#     Args:
#         results: YOLO 탐지 결과 (Ultralytics YOLO output list)
#         img_shape: (H, W) : 원본 이미지의 크기
#         conf_thresh: confidence threshold
#     Returns:
#         M_det_dict: {(b, class_id): sparse M_det tensor ∈ R^{HW x 3HW}}
#     """
#     H, W = img_shape
#     HW = H * W
#     CHW = 3 * HW

#     M_det_dict = {}

#     for b, result in enumerate(results):
#         if result.boxes is None or len(result.boxes) == 0:
#             continue

#         boxes = result.boxes.xyxy.cpu().numpy()
#         confs = result.boxes.conf.cpu().numpy()
#         classes = result.boxes.cls.cpu().numpy().astype(int)

#         img1_shape = (640, 640)
#         img0_shape = (H, W)
#         boxes = scale_coords(img1_shape, boxes.copy(), img0_shape)

#         for box, conf, cls_id in zip(boxes, confs, classes):
#             if conf < conf_thresh:
#                 continue

#             x1, y1, x2, y2 = map(int, box)
#             x1, y1 = max(0, x1), max(0, y1)
#             x2, y2 = min(W, x2), min(H, y2)

#             # ROI → row/col index 생성
#             row_idx_list = []
#             col_idx_list = []

#             for y in range(y1, y2):
#                 for x in range(x1, x2):
#                     flat_idx = y * W + x
#                     for c in range(3):
#                         row_idx = flat_idx
#                         col_idx = c * HW + flat_idx
#                         row_idx_list.append(row_idx)
#                         col_idx_list.append(col_idx)

#             if len(row_idx_list) == 0:
#                 continue

#             indices = torch.tensor([row_idx_list, col_idx_list], dtype=torch.long)
#             values = torch.ones(len(row_idx_list), dtype=torch.float)
            
#             """
#             .indices(): 희소 원소 위치 (위의 i)
#             .values(): 실제 원소 값 (위의 v)
#             .size(): 전체 텐서의 크기
#             """
            
#             M_det_sparse = torch.sparse_coo_tensor(
#                 indices=indices,
#                 values=values,
#                 size=(HW, CHW)
#             )

#             M_det_dict[(b, cls_id)] = M_det_sparse

#     return M_det_dict



# _DIAG_IDX_CACHE = {}

# def _get_diag_indices(n: int, device) -> torch.Tensor:
#     """
#     returns indices for a diagonal sparse matrix in COO:
#         idx = [[0,1,2,...,n-1],
#                [0,1,2,...,n-1]]  (shape: [2, n])
#     """
#     key = (n, torch.device(device).type)
#     t = _DIAG_IDX_CACHE.get(key, None)
#     if t is None:
#         ar = torch.arange(n, device=device, dtype=torch.long)
#         t = torch.stack([ar, ar], dim=0)  # [2, n]
#         _DIAG_IDX_CACHE[key] = t
#     return t



