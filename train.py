import sys
import os

import torch
import argparse
from datetime import datetime
import time

import torch.nn as nn
import torch
from tqdm import tqdm

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from dataloader import UJEDDataset
from model.model import DGU_Net
from utils.data_utils import collate_fn
from ultralytics import YOLO


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = "" ## Your root directory where download this code --- CHANGE THIS

def parse_args():
    parser = argparse.ArgumentParser(description='Underwater_detection')
    parser.add_argument('--root_dir', type=str, default='', help='root directory for this project')
    parser.add_argument('--train_dir', type=str, default='',help='Training dataset')
    parser.add_argument('--val_dir', type=str, default='',help='Validation dataset')
    parser.add_argument('--train_logs', type=str, default='',help='Training logs and outputs')
    parser.add_argument('--seed', type=int, default=5, metavar='S', help='random seed (default: 5)')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N', help='input batch size (default: 1)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='initial learning rate (default: 1e-4)')
    parser.add_argument('--en_weight', type=str, default=None, help='Resume training from saved checkpoint(s).',)
    parser.add_argument('--det_weight', type=str, default='', help='Path to YOLO detection model')
    parser.add_argument('--n_block', type=int, default=3, help='Number of layers in the Dual_Net model')
    args = parser.parse_args()
    
    return args

def main(args):
    
    ### 1. set up training logs
    args.train_logs = os.path.join(args.train_logs, datetime.now().strftime(f"%Y%m%d-%H%M%S"))
    os.makedirs(args.train_logs, exist_ok=True) 
    
    ### 2. dataset and dataloader
    transform =[transforms.ToTensor()]
    
    print("Training dataset loading...")
    train_dataloader = DataLoader(
        UJEDDataset(args.train_dir, transform=transform),
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=collate_fn,  
        num_workers=2,
    )
    print(f"{len(train_dataloader) * args.batch_size} images have been loaded")

    print("Testing dataset is loading...")
    test_dataloader = DataLoader(
        UJEDDataset(args.val_dir, transform=transform),
        batch_size=1,
        collate_fn=collate_fn,  
        num_workers=2,
    )
    print(f"{len(test_dataloader) } images have been loaded")
    
    ### 3. load network
    en_model = DGU_Net(Block_number = args.n_block).to(DEVICE)
    det_api = YOLO(args.det_weight)
    det_model = det_api.model.eval()            # pretrained detection model is used for extracting detected region for detection guided prior
    for p in det_model.parameters():
        p.requires_grad_(False)
    if hasattr(det_model, 'fuse'):
        det_model.fuse()
    det_model.to(DEVICE)
    
    
    ### 4. define loss function
    criterion = torch.nn.L1Loss().to(DEVICE)
    
    ### 5. optimizer
    optimizer = torch.optim.Adam(en_model.parameters(), lr=args.lr, betas = (0.9, 0.99))
    
    ### 6. resume (optional)
    start_epoch=1
    if args.en_weight is not None:
        print(f"Loading model from {args.en_weight}")
        checkpoint = torch.load(args.en_weight)
        en_model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    ### 7. train
    best_psnr = 0.0
    best_ssim = 0.0
    best_lpips = 2.0
    for epoch in tqdm(range(start_epoch, args.epochs+1)):
        en_model.train()
        s = time.time()
        for i,batch in enumerate(train_dataloader):
            input_img = batch["inp"].float().to(DEVICE)
            gt_img = batch["gt"].float().to(DEVICE)
            t_p = batch["t"].float().to(DEVICE)
            B_p = batch["B"].float().to(DEVICE)
            labels = batch["labels"].float().to(DEVICE) # labels are not used in this training
            if t_p.shape[1] == 1:
                t_p = t_p.repeat(1,3,1,1)
                
            print(f"input_img shape: {input_img.shape}, gt_img shape: {gt_img.shape}, t_p shape: {t_p.shape}, B_p shape: {B_p.shape}, labels shape: {labels.shape}")
            
            optimizer.zero_grad()
            output_J, output_t, output_B = en_model(input_img, t_p, B_p, labels, det_api)
            enhanced_img = output_J[-1]

            recon_loss = criterion(enhanced_img, gt_img)
            
            e = time.time()
            recon_loss.backward()
            optimizer.step()
             
            if not i % 10:
                sys.stdout.write(f"\rEpoch [{epoch}/{args.epochs}] Batch [{i+1}/{len(train_dataloader)}] Loss: {recon_loss.item():.4f} Time: {(e - s):.2f}s ")
                sys.stdout.flush() 
        
        ### 8. validate
        en_model.eval()
        val_loss = 0.
        with torch.inference_mode():      
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
                enhanced_img = output_J[-1]
                
                recon_loss = criterion(enhanced_img, gt_img)
                val_loss += recon_loss.item()
                
                batch_img_num = gt_img.size(0)
                num_imgs += batch_img_num

        print(f"Validation Loss: {val_loss/num_imgs:.4f}\n")
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