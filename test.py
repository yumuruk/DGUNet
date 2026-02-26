import sys
import os
import torch
import argparse
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataloader import UJEDDataset
from model.model import DGU_Net
from utils.data_utils import collate_fn
from ultralytics import YOLO

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description='Testing enhancement model')
    parser.add_argument('--test_dir', type=str, default='', help='Test dataset path')
    parser.add_argument('--save_dir', type=str, default='', help='Directory to save enhanced images')
    parser.add_argument('--image_size', type=int, help='Image size')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size')
    parser.add_argument('--n_block', type=int, default=3, help='Number of Dual_Net blocks')
    parser.add_argument('--en_weight', type=str, default='', help='Path to trained enhancement model')
    parser.add_argument('--det_weight', type=str, default='', help='Path to trained enhancement model')
    args = parser.parse_args()
    return args

def main(args):
    args.save_dir = os.path.join(args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    # DataLoader
    transform = [transforms.ToTensor()]
    test_dataset = UJEDDataset(args.test_dir, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    print(f"[INFO] Loaded {len(test_loader)} test batches.")

    # Model
    print("Loading Enhance Net...")
    model = DGU_Net(Block_number=args.n_block).to(DEVICE)
    checkpoint = torch.load(args.en_weight, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    det_api = YOLO(args.det_weight)

    det_model = det_api.model.eval()      
    for p in det_model.parameters():
        p.requires_grad_(False)
    det_model.to(DEVICE)
    

    # Inference
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_img = batch["inp"].to(DEVICE)   
            t_p = batch["t"].to(DEVICE)
            B_p = batch["B"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)   
            fn_list = batch["fn"]


            gt_img = batch.get("gt", None)
            if gt_img is not None:
                gt_img = gt_img.to(DEVICE)

            if t_p.shape[1] == 1:
                t_p = t_p.repeat(1, 3, 1, 1)

            output_J, _, _ = model(input_img, t_p, B_p, labels, det_api)
            enhanced_img = output_J[-1]

            # Save each image
            for idx in range(enhanced_img.size(0)):
                fn = fn_list[idx]
                save_path = os.path.join(args.save_dir, fn)
                save_image(enhanced_img[idx].clamp(0, 1), save_path)

    print("[DONE] All enhanced images saved.")
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
