import torch
from torchvision.datasets.coco import CocoDetection


class COCODataset(CocoDetection):
    """
    COCODataset is a class that inherits from torchvision.datasets.coco.CocoDetection,
    used for loading and preprocessing the COCO dataset.

    Parameters:
        ann_file (str): Path to the COCO annotation file.
        root (str): Root directory of the image files.
        transform (callable, optional): A function/transform that takes in an image and returns a transformed version.

    Methods:
        __getitem__(index: int) -> torch.Tuple[torch.Any]:
            Retrieves the image and its corresponding bounding boxes at the specified index.
    """
    def __init__(self, ann_file, root, transform=None):
        super().__init__(root, ann_file)
        self.transform = transform

    def __getitem__(self, index: int) -> torch.Tuple[torch.Any]:
        """
        Retrieves the image and its corresponding bounding boxes at the specified index.

        Parameters:
            index (int): Index in the dataset.

        Returns:
            img (torch.Tensor): The image torch.
            boxes (torch.Tensor): Normalized bounding boxes in the format [1, x_center, y_center, width, height].
        """
        img, anno = super().__getitem__(index)
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]

        boxes = []
        for ann in anno:
            x, y, w, h = ann['bbox']
            x_center = (x + w / 2) / img_info['width']
            y_center = (y + h / 2) / img_info['height']
            w = w / img_info['width']
            h = h / img_info['height']
            boxes.append([x_center, y_center, w, h])

        if len(boxes) == 0:
            boxes = [[0.5, 0.5, 0.1, 0.1]]

        boxes = torch.tensor(boxes, dtype=torch.float32)

        if self.transform:
            img = self.transform(img)

        return img, boxes
