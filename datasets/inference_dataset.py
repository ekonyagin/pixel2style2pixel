import cv2
from torch.utils.data import Dataset

from utils import data_utils


class InferenceDataset(Dataset):

    def __init__(self, root, opts, transform=None):
        self.paths = sorted(data_utils.make_dataset(root))
        self.transform = transform
        self.opts = opts

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from_path = self.paths[index]
        from_im = cv2.imread(from_path)
        from_im = cv2.cvtColor(from_im, cv2.COLOR_BGR2RGB)

        if self.transform:
            from_im = self.transform(from_im)
        return from_im
