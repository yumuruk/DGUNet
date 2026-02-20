import sys
import os
import torch
import argparse
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataloader import UIEBDataset
from model.model import Dual_Net
from utils.data_utils import collate_fn
from ultralytics import YOLO

from torchmetrics.image import PeakSignalNoiseRatio  # ✅ 추가

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description='Testing enhancement model')
    parser.add_argument('--test_dir', type=str, default='/media/ssd1/hansung/Dataset/UPRC_enhance_jpg/test', help='Test dataset path')
    parser.add_argument('--save_dir', type=str, default='./public/10103111_pretrained', help='Directory to save enhanced images')
    parser.add_argument('--image_size', type=int, help='Image size')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--n_block', type=int, default=3, help='Number of Dual_Net blocks')
    parser.add_argument('--en_weight', type=str, default='/media/ssd1/hansung/Code/UnderwaterDetection/weights/100_model.pth', help='Path to trained enhancement model')
    args = parser.parse_args()
    return args

def main(args):
    # args.save_dir = os.path.join(args.save_dir, args.en_weight.split('/')[-2], args.test_dir.split('/')[-1])
    args.save_dir = os.path.join(args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    # DataLoader
    transform = [transforms.ToTensor()]
    test_dataset = UIEBDataset(args.test_dir, transform=transform, img_size=args.image_size)
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
    model = Dual_Net(Block_number=args.n_block).to(DEVICE)
    checkpoint = torch.load(args.en_weight, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    det_api = YOLO('/media/ssd1/hansung/Code/UnderwaterDetection/weights/enhanced_9e.pt')  # 고수준 래퍼
    # print(f"det_api type : {type(det_api)},det_api.model type : {type(det_api.model)}")

    det_model = det_api.model.eval()            # 순수 nn.Module은 eval로 고정
    for p in det_model.parameters():
        p.requires_grad_(False)
    det_model.to(DEVICE)
    
    # ✅ PSNR metric
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)

    total_psnr = 0.0
    total_imgs = 0

    # Inference
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_img = batch["inp"].to(DEVICE)   # ToTensor면 보통 float32라 .float() 생략 가능
            t_p = batch["t"].to(DEVICE)
            B_p = batch["B"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)   # Not used
            fn_list = batch["fn"]

            # ✅ GT가 있는 경우에만 PSNR 계산
            gt_img = batch.get("gt", None)
            if gt_img is not None:
                gt_img = gt_img.to(DEVICE)

            if t_p.shape[1] == 1:
                t_p = t_p.repeat(1, 3, 1, 1)

            output_J, _, _ = model(input_img, t_p, B_p, labels, det_api)
            enhanced_img = output_J[-1]

            # ✅ PSNR 계산 (GT 있을 때)
            if gt_img is not None:
                # PSNR/SSIM/LPIPS 쪽은 보통 0~1 가정 → clamp 추천
                enh_c = enhanced_img.clamp(0, 1)
                gt_c = gt_img.clamp(0, 1)

                # batch-wise psnr (스칼라)
                batch_psnr = psnr_metric(enh_c, gt_c).item()
                bs = enh_c.size(0)

                total_psnr += batch_psnr * bs
                total_imgs += bs

            # Save each image
            for idx in range(enhanced_img.size(0)):
                fn = fn_list[idx]
                save_path = os.path.join(args.save_dir, fn)
                save_image(enhanced_img[idx].clamp(0, 1), save_path)

    if total_imgs > 0:
        print(f"[RESULT] Avg PSNR over {total_imgs} images: {total_psnr / total_imgs:.4f} dB")
    else:
        print("[WARN] GT not found in batches. PSNR was not computed.")

    print("[DONE] All enhanced images saved.")
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
