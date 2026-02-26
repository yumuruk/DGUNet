import torch
import torch.nn as nn

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