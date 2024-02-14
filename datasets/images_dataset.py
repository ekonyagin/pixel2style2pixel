from torch.utils.data import Dataset

import cv2
from utils import data_utils


class ImagesDataset(Dataset):

	TARGET_SIZE = 256

	def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None):
		self.source_paths = sorted(data_utils.make_dataset(source_root))
		self.target_paths = sorted(data_utils.make_dataset(target_root))
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.opts = opts

	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		from_path = self.source_paths[index]

		from_im = cv2.imread(from_path)
		from_im = cv2.cvtColor(from_im, cv2.COLOR_BGR2RGB)
		from_im = cv2.resize(from_im, (self.TARGET_SIZE, self.TARGET_SIZE), interpolation=cv2.INTER_LINEAR)

		to_path = self.target_paths[index]

		to_im = cv2.imread(to_path)
		to_im = cv2.cvtColor(to_im, cv2.COLOR_BGR2RGB)
		to_im = cv2.resize(to_im, (self.TARGET_SIZE, self.TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
		if self.target_transform:
			to_im = self.target_transform(to_im)

		if self.source_transform:
			from_im = self.source_transform(from_im)
		else:
			from_im = to_im

		return from_im, to_im
