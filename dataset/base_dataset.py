import lmdb
from torch.utils.data import Dataset
import torchvision.transforms as transforms


import os


class BaseDataset(Dataset):
    def __init__(self, logger):
        self.logger = logger
        self.env_img = None

        self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        return

    def __len__(self):
        return len(self.datalist)

    def open_lmdb_img(self):
        if self.use_lmdb and self.env_img is None:
            assert os.path.exists(self.root_lmdb), f"lmdb file not found: {self.root_lmdb}"
            self.env_img = lmdb.open(self.root_lmdb,readonly=True,lock=False,readahead=False,meminit=False, map_size=(1024)**3,max_spare_txns=32,max_dbs=5000)
        
    

