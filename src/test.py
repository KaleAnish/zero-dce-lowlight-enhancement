import torch
import torchvision
import os
import model
import numpy as np
from PIL import Image
import glob
import time

def lowlight(image_path, output_dir):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Load image and convert to RGB
    data_lowlight = Image.open(image_path).convert("RGB")
    data_lowlight = (np.asarray(data_lowlight) / 255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1).unsqueeze(0).cuda()

    # Load model
    DCE_net = model.enhance_net_nopool().cuda()
    DCE_net.load_state_dict(torch.load('snapshots_DARK/Epoch150.pth'))
    DCE_net.eval()

    # Inference
    start = time.time()
    _, enhanced_image, _ = DCE_net(data_lowlight)
    print(f"Time taken: {time.time() - start:.4f} seconds")

    # Save image with _enhanced suffix
    name, ext = os.path.splitext(os.path.basename(image_path))
    save_path = os.path.join(output_dir, f"{name}_enhanced{ext}")
    os.makedirs(output_dir, exist_ok=True)
    torchvision.utils.save_image(enhanced_image, save_path)
    print(f"Saved to: {save_path}")

if __name__ == '__main__':
    with torch.no_grad():
        # 🔧 SET THESE EXPLICITLY:
        input_folder = r"Zero-DCE_code\data\test_data\DARK"
        output_folder = r"C:\Users\aryan\Downloads\Zero-DCE-master\Zero-DCE_code\results"

        # Process each image
        supported_exts = ('.bmp', '.png', '.jpg', '.jpeg')
        for file in os.listdir(input_folder):
            if file.lower().endswith(supported_exts):
                input_path = os.path.join(input_folder, file)
                print(f"Processing: {input_path}")
                lowlight(input_path, output_folder)
