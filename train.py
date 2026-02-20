import sys
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import torch
import argparse
from datetime import datetime
import time
import random
import numpy as np

import torch.nn as nn
import torch
from tqdm import tqdm

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import pyiqa

from dataloader import UIEBDataset
from model.model import Dual_Net
from utils.data_utils import collate_fn
from ultralytics import YOLO


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description='Underwater_detection')
    parser.add_argument('--train_dir', type=str, default='/media/ssd1/hansung/Dataset/UPRC_enhance_jpg/train',help='Training dataset')
    parser.add_argument('--val_dir', type=str, default='/media/ssd1/hansung/Dataset/UPRC_enhance_jpg/valid',help='Validation dataset')
    parser.add_argument('--train_logs', type=str, default='./train_logs',help='Training logs and outputs')
    parser.add_argument('--seed', type=int, default=5, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N', help='input batch size (default: 1)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='initial learning rate (default: 1e-4)')
    # parser.add_argument('--n_negative_samples', type=int, default=3, metavar='N', help='number of negative samples to train (default: 5)') # for contrastive learning
    parser.add_argument('--en_weight', type=str, default=None, help='Resume training from saved checkpoint(s).',)
    parser.add_argument('--image_size', type=int, default=416, help='Image size for training and testing')
    parser.add_argument('--n_block', type=int, default=3, help='Number of layers in the Dual_Net model')
    args = parser.parse_args()
    
    return args

def main(args):
    
   
    args.train_logs = os.path.join(args.train_logs, datetime.now().strftime(f"%Y%m%d-%H%M%S_seed{args.seed}_131111"))
    os.makedirs(args.train_logs, exist_ok=True) 
    ### 1. visdom (optional)
    
    ### 2. dataset and dataloader
    transform =[
        # transforms.Resize((args.image_size, args.image_size)),
        # transforms.Resize((480, 360)),
        transforms.ToTensor()
    ]
    
    print("Training dataset loading...")
    train_dataloader = DataLoader(
        UIEBDataset(args.train_dir, transform=transform, img_size=args.image_size),
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=collate_fn,  
        num_workers=2,
    )
    print(f"{len(train_dataloader) * args.batch_size} images have been loaded")

    print("Testing dataset is loading...")
    test_dataloader = DataLoader(
        UIEBDataset(args.val_dir, transform=transform, img_size=args.image_size), 
        batch_size=1,
        collate_fn=collate_fn,  
        num_workers=2,
    )
    print(f"{len(test_dataloader) } images have been loaded")
    
    ### 3. network
    en_model = Dual_Net(Block_number = args.n_block).to(DEVICE)
    det_api = YOLO('./weights/enhanced_9e.pt')  # 고수준 래퍼
    det_model = det_api.model.eval()            # 순수 nn.Module은 eval로 고정
    for p in det_model.parameters():
        p.requires_grad_(False)
    if hasattr(det_model, 'fuse'):
        det_model.fuse()
    det_model.to(DEVICE)
    
    
    ### 4. loss && metrics
    criterion = torch.nn.L1Loss().to(DEVICE)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    lpips = pyiqa.create_metric('lpips', model='alex', normalize=True, data_range=1.0).to(DEVICE)
    
    ### 5. optimizer
    optimizer = torch.optim.Adam(en_model.parameters(), lr=args.lr, betas = (0.9, 0.99))
    
    ### 6. scheduler
    # None scheduler for enhancement network
    
    ### 7. resume 
    start_epoch=1
    if args.en_weight is not None:
        print(f"Loading model from {args.en_weight}")
        checkpoint = torch.load(args.en_weight)
        en_model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # due to equal learning rate
        start_epoch = checkpoint['epoch'] + 1
    
    ### 8. train
    best_psnr = 0.0
    best_ssim = 0.0
    best_lpips = 2.0
    for epoch in tqdm(range(start_epoch, args.epochs+1)):
        en_model.train()
        # en_model.zero_grad()
        s = time.time()
        for i,batch in enumerate(train_dataloader):
            input_img = batch["inp"].float().to(DEVICE)
            gt_img = batch["gt"].float().to(DEVICE)
            t_p = batch["t"].float().to(DEVICE)
            B_p = batch["B"].float().to(DEVICE)
            labels = batch["labels"].float().to(DEVICE) # labels are not used in this training
            if t_p.shape[1] == 1:
                t_p = t_p.repeat(1,3,1,1)
            
            optimizer.zero_grad()
            # print(f"label tensor shape: {labels.shape}")
            output_J, output_t, output_B = en_model(input_img, t_p, B_p, labels, det_api)
            # output_J, output_t, output_B = en_model(input_img, t_p, B_p)
            enhanced_img = output_J[-1]
            # print(f"enhanced_image val max :{enhanced_img.max().item()}, min: {enhanced_img.min().item()}")
            # print(f"gt_image val max :{gt_img.max().item()}, min: {gt_img.min().item()}")

            recon_loss = criterion(enhanced_img, gt_img)
            
            e = time.time()
            recon_loss.backward()
            optimizer.step()
             
            if not i % 10:
                sys.stdout.write(f"\rEpoch [{epoch}/{args.epochs}] Batch [{i+1}/{len(train_dataloader)}] Loss: {recon_loss.item():.4f} Time: {(e - s):.2f}s ")
                sys.stdout.flush() 
                
        # save_path = os.path.join(args.train_logs, f"model_epoch_{epoch}.pth")
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': en_model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        # }, save_path)
        
        ### 9. validate
        en_model.eval()
        val_loss = 0.
        with torch.inference_mode():
            
            psnr_val, ssim_val, lpips_val = 0.0, 0.0, 0.0
            clamp_psnr, clamp_ssim, clamp_lpips = 0.0, 0.0, 0.0
            num_imgs = 0 
            
            for i, batch in enumerate(test_dataloader):
                input_img = batch["inp"].float().to(DEVICE)
                gt_img = batch["gt"].float().to(DEVICE)
                t_p = batch["t"].float().to(DEVICE)
                B_p = batch["B"].float().to(DEVICE)
                labels = batch["labels"].float().to(DEVICE)
                if t_p.shape[1] == 1:
                    t_p = t_p.repeat(1,3,1,1)

                output_J, output_t, output_B = en_model(input_img, t_p, B_p, labels, det_api)
                # output_J, output_t, output_B = en_model(input_img, t_p, B_p)
                enhanced_img = output_J[-1]
                
                recon_loss = criterion(enhanced_img, gt_img)
                val_loss += recon_loss.item()
                
                batch_img_num = gt_img.size(0)
                num_imgs += batch_img_num
                
                # psnr_val += psnr(enhanced_img, gt_img).item()
                # ssim_val += ssim(enhanced_img, gt_img).item()
                # lpips_val += lpips(enhanced_img, gt_img).item()
                
                enhanced_img = enhanced_img.clamp(0, 1)
                gt_img = gt_img.clamp(0, 1)
                
                clamp_psnr += psnr(enhanced_img, gt_img).item()
                clamp_ssim += ssim(enhanced_img, gt_img).item()
                clamp_lpips += lpips(enhanced_img, gt_img).item()

        print(f"Validation Loss: {val_loss/num_imgs:.4f}, PSNR: {psnr_val/num_imgs:.4f}, SSIM: {ssim_val/num_imgs:.4f}\n")
        print(f"Clamped PSNR: {clamp_psnr/num_imgs:.4f}, Clamped SSIM: {clamp_ssim/num_imgs:.4f}, Clamped LPIPS: {clamp_lpips/num_imgs:.4f}\n")
        if clamp_psnr / num_imgs > best_psnr:
            best_psnr = clamp_psnr / num_imgs
            torch.save({
                'epoch': epoch,
                'model_state_dict': en_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.train_logs, f"best_psnr_model.pth"))
        if clamp_ssim / num_imgs > best_ssim:
            best_ssim = clamp_ssim / num_imgs
            torch.save({
                'epoch': epoch,
                'model_state_dict': en_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.train_logs, f"best_ssim_model.pth"))
        if clamp_lpips / num_imgs < best_lpips:
            best_lpips = clamp_lpips / num_imgs
            torch.save({
                'epoch': epoch,
                'model_state_dict': en_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.train_logs, f"best_lpips_model.pth"))
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': en_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.train_logs, f"model_epoch_{epoch}.pth"))
    print(f"Best PSNR: {best_psnr:.4f}, Best SSIM: {best_ssim:.4f}, Best LPIPS: {best_lpips:.4f}\n")        

if __name__ == "__main__":
    args = parse_args()
    main(args)