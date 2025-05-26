from ultralytics import YOLO
from ultralytics.utils import IterableSimpleNamespace
import torch


def get_grad_direction(img: torch.Tensor, model) -> torch.Tensor:
    model.eval()
    model.args = IterableSimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    train_batch = {
        'cls': torch.zeros((0, 1)),
        'bboxes': torch.zeros((0, 4)),
        'batch_idx': torch.zeros(0)
    }
    img.requires_grad_()
    loss, _ = model.loss(train_batch, model(img))
    loss.backward()
    loss.grad = None
    return torch.sign(img.grad)


def generate_perturbation(
    x_origin: torch.Tensor,
    eps=8/255,
    lr=2/255,
    epochs=10
) -> tuple[torch.Tensor, torch.Tensor]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_origin = x_origin.to(device)
    x_adv = x_origin.clone().to(device)
    noise = (
        torch.rand(x_adv.size(), dtype=torch.float).to(device) - .5
        ) * eps * 2
    for _ in range(epochs):
        grad_direction = get_grad_direction(
            x_adv, YOLO("yolo11n.pt").model.to(device)
        ).to(device)
        noise = torch.clip(noise - lr * grad_direction, -eps, eps)
        x_adv = torch.clip(x_origin + noise, 0, 1)
    return x_adv, noise
