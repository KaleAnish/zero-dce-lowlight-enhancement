import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import Myloss
import numpy as np
from torchvision import transforms


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    DCE_net = model.enhance_net_nopool().cuda()
    DCE_net.apply(weights_init)

    if config.load_pretrain:
        print("✅ Loading pretrained weights from:", config.pretrain_dir)
        DCE_net.load_state_dict(torch.load(config.pretrain_dir))

    train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    L_color = Myloss.L_color()
    L_spa = Myloss.L_spa()
    L_exp = Myloss.L_exp(16, 0.6)
    L_TV = Myloss.L_TV()

    optimizer = torch.optim.Adam(DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    DCE_net.train()
    start_time = time.time()

    for epoch in range(config.num_epochs):
        epoch_start = time.time()

        for iteration, img_lowlight in enumerate(train_loader):
            img_lowlight = img_lowlight.cuda()

            enhanced_image_1, enhanced_image, A = DCE_net(img_lowlight)

            Loss_TV = 200 * L_TV(A)
            loss_spa = torch.mean(L_spa(enhanced_image, img_lowlight))
            loss_col = 5 * torch.mean(L_color(enhanced_image))
            loss_exp = 10 * torch.mean(L_exp(enhanced_image))

            loss = Loss_TV + loss_spa + loss_col + loss_exp

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(DCE_net.parameters(), config.grad_clip_norm)
            optimizer.step()

            if (iteration + 1) % config.display_iter == 0:
                print(f"Epoch {epoch+1} | Iteration {iteration+1} | Loss: {loss.item():.4f}")

        # Save model at end of epoch
        save_path = os.path.join(config.snapshots_folder, f"Epoch{epoch+1}.pth")
        try:
            torch.save(DCE_net.state_dict(), save_path)
            print(f"✔ Model saved at: {save_path}")
        except Exception as e:
            print(f"Failed to save model at epoch {epoch+1}: {e}")

        print(f"Epoch {epoch+1} completed in {time.time() - epoch_start:.2f} seconds")

    print(f"\n Training completed in {(time.time() - start_time)/60:.2f} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--lowlight_images_path', type=str, default="C:/Users/aryan/Downloads/Zero-DCE-master/Zero-DCE_code/data/LOL_Dataset")

    # Optimization-related hyperparameters
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)

    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=10)

    # Console and saving
    parser.add_argument('--display_iter', type=int, default=20)

    # Model checkpointing
    parser.add_argument('--snapshots_folder', type=str, default="C:/Users/aryan/Downloads/Zero-DCE-master/snapshots_LOL/")
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_dir', type=str, default="C:/Users/aryan/Downloads/Zero-DCE-master/snapshots_LOL/Epoch200.pth")

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)

    train(config)
