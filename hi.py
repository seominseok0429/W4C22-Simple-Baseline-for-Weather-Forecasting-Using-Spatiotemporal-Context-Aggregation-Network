import random
import numpy as np
import torch
from skimage.transform import resize
from torch.utils.data import DataLoader

class GK2A(object):
    def __init__(self, data_root, resize_width, is_train):
        self.path = data_root
        self.data = np.load(self.path, allow_pickle=True)
        self.image_width = np.shape(np.load(self.data[0][0][0]))[0]
        self.resize_width = resize_width
        self.min_ir = 200
        self.max_ir = 300
        self.is_train = is_train
          
    def __len__(self):
        return np.shape(self.data)[0]

    def __getitem__(self, idx):
        conc_img_seq = []
        mul_ch_seq = self.data[idx]
        for ch_idx, sg_ch_seq in enumerate(mul_ch_seq):
            if ch_idx < 2:
                # divide class
                img_seq = [np.load(s) for s in sg_ch_seq]
                for idx, npy in enumerate(img_seq):
                    npy = npy[:, 200:-200]  ## 2600*2600 crop
                    if self.resize_width > 0:
                        if ch_idx == 0:  # CLD
                            npy = resize(npy, (self.resize_width, self.resize_width), anti_aliasing=False, preserve_range=True)
                            npy = np.where((npy > 0) & (npy < 2), 1, npy)  
                            # (resize하고나면 class1의 값이 소수점 자리를 가진 이 범위의 값이 됨) True -> 1, False -> 원래값
                        else:
                            npy = resize(npy,(self.resize_width, self.resize_width), anti_aliasing=True, preserve_range=True)
                    npy[np.isnan(npy)] = np.nanmin(npy)
                    img_seq[idx] = npy
                img_seq = np.stack(img_seq)
                img_seq = torch.Tensor(img_seq)

                # if ch_idx == 0:  ## CLD
                #     img_seq = label_to_one_hot_label(img_seq, 3)
                # elif ch_idx == 1:  ## IR
                
                if ch_idx == 1:  ## IR
                    img_seq = (img_seq - self.min_ir) / (self.max_ir - self.min_ir)
                    img_seq = torch.clamp(img_seq, 0, 1)  # 의미없는 값 제거

                # if len(mul_ch_seq) > 1:
                #     if ch_idx != 0:
                #         img_seq = img_seq.unsqueeze(3)

                
                if len(mul_ch_seq) > 1:
                    img_seq = img_seq.unsqueeze(3)
                        
                if ch_idx == 0:
                    conc_img_seq = img_seq
                else:
                    conc_img_seq = torch.cat([conc_img_seq, img_seq], dim = 3)
            
        # (t, h, w, c) -> (c, t, w, h)=(2, 20, 128, 128)
        conc_img_seq = conc_img_seq.permute(3,0,2,1)
        # inputs, targets = conc_img_seq[:, :10, :, :], conc_img_seq[:1, 10:, :, :]
        inputs, targets = conc_img_seq[:, :10, :, :], conc_img_seq[0, 10:, :, :]

        return inputs, targets  # (2, 10, 128, 128), (10, 128, 128)

if __name__ == "__main__":
  
    def find_indices(image_tensor):
        mask = (image_tensor != 0) & (image_tensor != 1) & (image_tensor != 2)
        indices = torch.where(mask)
        return indices
    train_dataset = GK2A(data_root='/scratch/q593a18/workspace/PROJECTS/caf/output/gk2a_2020_20len_30min_org_CLD_IR105_WV063.npy', 
                                   resize_width=128,
                                   is_train=True)
    val_dataset = GK2A(data_root='/scratch/q593a18/workspace/PROJECTS/caf/output/gk2a_2021_20len_30min_org_CLD_IR105_WV063.npy', 
                                   resize_width=128,
                                   is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, 
                              num_workers=16, pin_memory=True, prefetch_factor=2, 
                              persistent_workers=False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, 
                            num_workers=16, pin_memory=True, prefetch_factor=2, 
                            persistent_workers=False, drop_last=True)

    for idx, i in enumerate(train_dataset):
        print(find_indices(i[1]))
