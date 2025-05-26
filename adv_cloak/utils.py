import torch
from patch.patch_gen import TotalVariation


def multi_detection_loss(output, patch):
    cls_loss = torch.zeros(80, dtype=torch.float32).to(patch.device)

    for i in range(80):
        for result in output:
            boxes = result.boxes[result.boxes.cls == i]
            if len(boxes) > 0:
                cls_loss[i] += boxes.conf.mean()

    color_loss = -torch.mean(torch.max(patch, dim=0)[0] - torch.min(patch, dim=0)[0])
    grey_loss = -torch.mean(torch.std(patch, dim=0))
    tv_loss = TotalVariation()
    tv = tv_loss(patch)

    loss = torch.norm(cls_loss) / 80 + 0.05 * tv + 0.2 * color_loss + 0.2 * grey_loss
    return loss
