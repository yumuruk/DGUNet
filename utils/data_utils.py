import os
import torch

def collate_fn(batch):
    # Separate images, labels, and other data
    inps = torch.stack([item['inp'] for item in batch], 0)
    gts = torch.stack([item['gt'] for item in batch], 0)
    ts = torch.stack([item['t'] for item in batch], 0)
    Bs = torch.stack([item['B'] for item in batch], 0)
    fn = [item['fn'] for item in batch]  # <- 파일 이름 리스트 유지
    
    # Process labels
    labels = []
    for i, item in enumerate(batch):
        label = item['labels']
        if label.shape[0] > 0:
            # Add batch index as the first column
            # label shape: [n_obj, 5] -> [n_obj, 6] (batch_idx, class, x, y, w, h)
            batch_idx = torch.full((label.shape[0], 1), i)
            label = torch.cat((batch_idx, label), 1)
            labels.append(label)
            
    if len(labels) > 0:
        labels = torch.cat(labels, 0)
    else:
        # If no labels in the entire batch, create an empty tensor
        labels = torch.empty((0, 6))

    # return {'inp': inps, 'gt': gts, 't': ts, 'B': Bs, 'labels': labels}
    return {'inp': inps, 'gt': gts, 't': ts, 'B': Bs, 'labels': labels, 'fn': fn}