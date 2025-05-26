from ultralytics.models import yolo
from generate_perturbation import generate_perturbation
import torch


class TOG_DetectionValidator(yolo.detect.DetectionValidator):
    def preprocess(self, batch):
        batch = super().preprocess(batch)
        for i in range(len(batch["img"])):
            with torch.inference_mode(False):
                im = batch["img"][i].unsqueeze(0)
                im, _ = generate_perturbation(im)
            batch["img"][i] = im[0]
        return batch


class TOG_DetectionPredictor(yolo.detect.DetectionPredictor):
    def preprocess(self, im):
        im = super().preprocess(im)
        # Modification for TOG-Vanishing Attack in predicting
        with torch.inference_mode(False):
            im, _ = generate_perturbation(im)
        return im
