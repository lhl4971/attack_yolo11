import torch
import torch.nn.functional as F
from ultralytics import YOLO
from patch.patch_gen import PatchApplier
import os
import cv2
import numpy as np
from torchvision.utils import save_image


class PatchTrainer:
    def __init__(self, model, patch, criterion, optimizer, device="cpu"):
        self.model = YOLO(model).to(device)
        self.class_names = self.model.names
        self.adv_patch = patch
        self.criterion = criterion
        self.optimizer = optimizer
        self.patch_applier = PatchApplier().to(device)
        self.device = device

    def train(self, num_epochs, train_loader, val_loader=None):
        for epoch in range(num_epochs):
            running_loss = 0
            for index, data in enumerate(train_loader):
                images, targets = data
                images = torch.stack(images).to(self.device)

                # Detect object without attacks
                with torch.no_grad():
                    output_origin = self.model(images)

                self.optimizer.zero_grad()
                batch_patch = torch.zeros(images.shape, dtype=torch.float32).to(self.device)
                for i in range(len(images)):
                    img_patch = torch.zeros(images.shape[1:], dtype=torch.float32).to(self.device)

                    for x_center, y_center, width, height in targets[i]:
                        img_h, img_w = images.size(2), images.size(3)
                        x1 = int((x_center - width / 2) * img_w)
                        y1 = int((y_center - height / 2) * img_h)
                        x2 = int((x_center + width / 2) * img_w)
                        y2 = int((y_center + height / 2) * img_h)

                        # Calculate the target size of the patch.
                        target_height = y2 - y1
                        target_width = x2 - x1

                        # Make sure to leave at least 5% clearance on each side.
                        min_margin = 0.05
                        max_height = int(target_height * (1 - 2 * min_margin))
                        max_width = int(target_width * (1 - 2 * min_margin))

                        # Calculate resize ratio, keeping aspect ratio unchanged.
                        aspect_ratio = self.adv_patch.size(1) / self.adv_patch.size(2)
                        if max_height / aspect_ratio <= max_width:
                            new_height = max_height
                            new_width = int(max_height / aspect_ratio)
                        else:
                            new_width = max_width
                            new_height = int(max_width * aspect_ratio)

                        # If the resized patch is too large, only use the 100x150 patch.
                        max_patch_size = 150
                        if new_height > max_patch_size or new_width > max_patch_size:
                            if new_height > new_width:
                                new_height = max_patch_size
                                new_width = int(max_patch_size / aspect_ratio)
                            else:
                                new_width = max_patch_size
                                new_height = int(max_patch_size * aspect_ratio)

                        # Resize adv_patch.
                        resized_patch = F.interpolate(self.adv_patch.unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False).squeeze(0)

                        # Calculate paste position.
                        paste_y1 = y1 + (target_height - new_height) // 2
                        paste_x1 = x1 + (target_width - new_width) // 2

                        img_patch[:, paste_y1: paste_y1 + new_height, paste_x1:paste_x1 + new_width] = resized_patch
                    batch_patch[i] = img_patch

                input = self.patch_applier(images, batch_patch)
                output = self.model(input)
                loss = self.criterion(output, self.adv_patch)
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    self.adv_patch.clamp_(0, 1)

                    max_vals, _ = torch.max(self.adv_patch, dim=0)
                    min_vals, _ = torch.min(self.adv_patch, dim=0)
                    diff = max_vals - min_vals
                    mask = diff < 0.3
                    if mask.any():
                        max_channel = torch.argmax(self.adv_patch[:, mask], dim=0)
                        min_channel = torch.argmin(self.adv_patch[:, mask], dim=0)
                        self.adv_patch[max_channel, mask] *= 1.1
                        self.adv_patch[min_channel, mask] *= 0.9
                        self.adv_patch.clamp_(0, 1)

                running_loss += loss
                if index % 12 == 0:
                    self.save_images(epoch, images, input, output_origin, output)

            self.optimizer.step()
            avg_running_loss = running_loss / len(train_loader)
            print("Epoch: {}\tLoss:{:.4f}".format(epoch, avg_running_loss))

    def save_images(self, epoch, original_images, patched_images, original_results, patched_results):
        output_dir = os.path.join(f"./output/epoch_{epoch}")
        os.makedirs(output_dir, exist_ok=True)

        # Save the patch
        save_image(self.adv_patch, os.path.join(output_dir, 'patch.png'))
        for idx, (original_image, patched_image, original_result, patched_result) in enumerate(zip(original_images, patched_images, original_results, patched_results)):
            # Convert tensors to numpy arrays
            original_image = (original_image.cpu().detach().permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
            patched_image = (patched_image.cpu().detach().permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
            # Draw bounding boxes on original image
            for box in original_result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                label = f'{self.class_names[int(box.cls[0].cpu().numpy())]} {box.conf[0].cpu().numpy():.2f}'
                cv2.rectangle(original_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(original_image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            # Save patched_image image
                cv2.imwrite(os.path.join(output_dir, f'original_image_{idx}.png'), cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))

            # Draw bounding boxes on patched image
            for box in patched_result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                label = f'{self.class_names[int(box.cls[0].cpu().numpy())]} {box.conf[0].cpu().numpy():.2f}'
                cv2.rectangle(patched_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(patched_image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Save patched_image image
            cv2.imwrite(os.path.join(output_dir, f'patched_image_{idx}.png'), cv2.cvtColor(patched_image, cv2.COLOR_RGB2BGR))
