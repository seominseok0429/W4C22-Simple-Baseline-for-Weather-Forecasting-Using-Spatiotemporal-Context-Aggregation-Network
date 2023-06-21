from itertools import chain
import torch
import os
import gzip
import numpy as np
# from tqdm import tqdm
from pyproj import Proj
import cv2
import gc

import torch.nn.functional as F

class RainFDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, years, modalities, input_seq, img_size=256, cls_num=3, after=1, sampling = None, normalize_hsr = False, use_meta=True):
        print("Predict  IDX")
        self.img_size = img_size
        self.cls_num = cls_num
        self.use_meta = use_meta
        # print(modalities)
        modalities = modalities.copy()
        if 'radar' in modalities:
            modalities.remove('radar')
        self.modalities = modalities
        self.data_interval = 1
        self.date = []
        self.days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        self.dates = list(chain.from_iterable([chain.from_iterable([map(str, np.arange(year * 10000 + (month + 1) * 100 + 1 , year * 10000 + (month + 1) * 100 + 1 + self.days[month] + int((month == 1) and (year % 4 == 0))))for month in range(12)])
                                               for year in years]))
        self.inv_dates = {v: i for i, v in enumerate(self.dates)}
        self.after, self.input_dim = after, input_seq
        self.data_path = data_path
        self.radar_indices = []
        self.normalize_hsr = normalize_hsr

        for i in years:
            cur_path = data_path +f'/radar_{i}/'
            data_list = os.listdir(cur_path)
            data_list.sort()
            # print(self.inv_dates)
            print("# of Datas in radar :"+str(len(data_list)))
            for n in data_list:
                timestamp = n.split("-")[0]
                # print(timestamp)
                date, hour, minu = timestamp[:-4], int(timestamp[-4:-2]), int(timestamp[-2:])
                # if minu % 5 != 0:
                #     print(timestamp)
                #     continue
                idx = self.inv_dates[date] * 24 + hour

                self.radar_indices.append(int(idx)) # radar indices : yyyymmddHH 를 �| �| ��~U~\ idx �~H��~^~P

        self.radar_indices = list(set(self.radar_indices))
        print("# of Radars:"+str(len(self.radar_indices)))

        self.gt_indices= []
        gt_indices_inv_list= dict()
        gt_indices_inv1 = []

        for raw_idx in range(len(self.radar_indices)):
            check = True
            if raw_idx < input_seq -1 + after//self.data_interval:
                continue

            target_date_idx = self.radar_indices[raw_idx]
            for bef in range(1, 6):
                datestr = self.get_datestr(target_date_idx - bef)
                # print(datestr)
                if not os.path.exists(
                    f'{self.data_path}/radar_{datestr[0:4]}/{datestr}-radar.npy'):
                    check = False
                    break
                if not os.path.exists(f'{self.data_path}/hima_01/{datestr}-hima.npy'):
                    check = False
                    break
                if not os.path.exists(f'{self.data_path}/hima_06/{datestr}-hima.npy'):
                    check = False
                    break
            if check == False:
                continue

            # if sampling == None:
            #     if not self.get_datestr(target_date_idx).endswith("00"):
            #         continue
            for inv in range(1, 6+1):
                for i in range(self.input_dim):
                    datestr = self.get_datestr(target_date_idx - ((self.input_dim-1) + (inv * (self.after))) + i)
                    if not os.path.exists(f'{self.data_path}/radar_{datestr[0:4]}/{datestr}-radar.npy'):
                        check = False
                        break
                    if not os.path.exists(f'{self.data_path}/hima_01/{datestr}-hima.npy'):
                        check = False
                        break
                    if not os.path.exists(f'{self.data_path}/hima_06/{datestr}-hima.npy'):
                        check = False
                        break
                if check == True:
                    if sampling == None:
                        self.gt_indices.append((target_date_idx, inv))
                    else:
                        if inv == 1:
                            gt_indices_inv1.append(target_date_idx)
                        if target_date_idx not in gt_indices_inv_list:
                            gt_indices_inv_list[target_date_idx] = [inv]
                        else:
                            gt_indices_inv_list[target_date_idx].append(inv)

        if sampling != None:
            jump = 10//(int)(sampling*10)
            gt_indices_inv1 = gt_indices_inv1[0::jump]
#             print(str(len(gt_indices_inv1)))

            for trg_date_idx in gt_indices_inv1:
                if trg_date_idx in gt_indices_inv_list:
                    for _inv in gt_indices_inv_list[trg_date_idx]:
                        self.gt_indices.append((trg_date_idx, _inv))


        print("# of Ground Truths:"+str(len(self.gt_indices)))

        self.img_x = self.img_size
        self.img_y = self.img_size

        minlon=126.3
        maxlon=129.3
        minlat=35
        maxlat=38

        geo_lon, geo_lat = np.mgrid[minlon:maxlon:complex(0,self.img_size),minlat:maxlat:complex(0, self.img_size)]
        self.geo_lon = geo_lon.reshape(1,self.img_size,-1)
        self.geo_lat = geo_lat.reshape(1,self.img_size,-1)

        # for modality in self.modalities:
        #     temp = modality.replace()



    def get_datestr(self, _idx):
        # date = self.dates[int(_idx // 24)]
        # hour = int((_idx % 24) // 1)
        # minu = 0

        return "%s%02d00" % (self.dates[int(_idx // 24)], int(_idx % 24) )
        # else:
        #     return "%s%02d%2d" % (self.dates[int(_idx // 144)], int((_idx % 144) // 6), int((_idx % 6)*self.data_interval))

    def get_inv_idx(self, pred_date):
        date, hour, minu = pred_date[:-4], int(pred_date[-4:-2]), int(pred_date[-2:])
        return self.inv_dates[date] * 24 + hour

    def __len__(self) -> int:
        return len(self.gt_indices)

    def zr_relation(self, _radar):
        x = torch.FloatTensor(10**(_radar*0.1))
        return torch.FloatTensor((x/148)**(100/159))

    def __getitem__(self, raw_idx):
        def get_dBz(_idx):
            datestr = self.get_datestr(_idx)
            data = np.load(f'{self.data_path}/radar_{datestr[0:4]}/{datestr}-radar.npy', allow_pickle=True)
            # data = cv2.resize(data, (self.img_size, self.img_size))
            dBz = torch.tensor(data, dtype = torch.float)

            return dBz

        def parse(_idx, only_image):
                dBz = get_dBz(_idx)

                if only_image:
                    if self.normalize_hsr == True:
                        dBz = self.zr_relation(dBz)
                        return torch.tanh(torch.log(dBz+0.01)/4)
                    else:
                        return dBz

                if not only_image:
                    # for bef in range(1,6):
                    #     dBz = dBz + get_dBz(_idx - bef)
                    # dBz = dBz/6.
                    mask = torch.tensor(dBz >= -250)

                    return self.zr_relation(dBz), mask

        def get_sat(_idx):
            datestr = self.get_datestr(_idx)
            data = np.load(f'{self.data_path}/hima_01/{datestr}-hima.npy', allow_pickle=True)
            # data = data * 10
            # data = data.astype(np.int16)
            # data = cv2.resize(data, (960,960))

            # data = data/10
            # data = data.astype(np.float32)
            # data = np.nan_to_num(data)
            data = np.clip(data, 190., 300.)

            data = (data-190.)/110.
            # data = 1-data

            return data


        def get_sat2(_idx):
            datestr = self.get_datestr(_idx)
            data = np.load(f'{self.data_path}/hima_06/{datestr}-hima.npy', allow_pickle=True)
            # data = data * 10
            # data = data.astype(np.int16)
            # data = cv2.resize(data, (960,960))

            # data = data/10
            # data = data.astype(np.float32)
            # data = np.nan_to_num(data)
            data = np.clip(data, 190., 260.)

            data = (data-190.)/70.
            # data = 1-data

            return data

        def rainf(_idx, data):
            time = self.get_datestr(_idx)
            # print(time)
            # print('data:',data)

            if data == 'hima01':
                img = get_sat(_idx)

            elif data == 'hima06':
                img = get_sat2(_idx)


            elif data == 'imerg':
                path = f'{self.data_path}/imerg/{time}-imerg.npy'
                img = np.load(path, allow_pickle=True)
                # img = np.nan_to_num(img)
                img[np.where(img<0)] = 0
                # data = self.zr_relation(data)
                img = torch.tanh(torch.log(torch.FloatTensor(img+0.01))/4)

            elif data == 'rain':
                path = f'{self.data_path}/rain/{time}-Rain.npy'
                img = np.load(path, allow_pickle=True)
                # img = np.nan_to_num(img)
                img[np.where(img<0)] = 0
                img = torch.tanh(torch.log(torch.FloatTensor(img+0.01))/4)

            elif data == 'temp':
                path = f'{self.data_path}/temp/{time}-Temp.npy'
                img = np.load(path, allow_pickle=True)
                # img = np.nan_to_num(img)
                img = img/41.

            elif data == 'wdir':
                path = f'{self.data_path}/wdir/{time}-Wdir.npy'
                img = np.load(path, allow_pickle=True)
                # img = np.nan_to_num(img)
                img = img/360.


            elif data == 'wsp':
                path = f'{self.data_path}/wsp/{time}-Wsp.npy'
                img = np.load(path, allow_pickle=True)
                # img = np.nan_to_num(img)
                img = np.clip(img, 0., 20.)
                img = img/20.

            elif data == 'hum':
                path = f'{self.data_path}/hum/{time}-Hum.npy'
                img = np.load(path, allow_pickle=True)
                # img = np.nan_to_num(img)
                img = np.clip(img, 0., 100.)
                img = img/100.
            if data == 'wdir':
                pos_sin = np.sin(img*2*np.pi)
                pos_cos = np.cos(img*2*np.pi)
                img = np.stack([pos_sin, pos_cos], axis=0)
            return torch.Tensor(img).unsqueeze(0)

        predict_date_idx, lead = self.gt_indices[raw_idx]
        radar_history = torch.FloatTensor(np.stack([
            parse(predict_date_idx - ((self.input_dim - 1) + (lead * (self.after // self.data_interval))) + i, only_image = True) for i in range(self.input_dim)], axis=0))
        radar_history = radar_history.unsqueeze(1)
        concat_list = [radar_history]

        for data in self.modalities:
            if data == 'wdir':
                orther = torch.FloatTensor(
                    np.concatenate([rainf(predict_date_idx - ((self.input_dim - 1) + (lead * (self.after // self.data_interval))) + i, data) for i in range(self.input_dim)], axis=0))
                concat_list.append(orther)
            else:
                orther = torch.FloatTensor(
                        np.stack([rainf(predict_date_idx - ((self.input_dim - 1) + (lead * (self.after // self.data_interval))) + i, data) for i in range(self.input_dim)], axis=0))
                concat_list.append(orther)
        radar_history = torch.cat(concat_list, axis=1)

        if self.use_meta:
            cur_datestr = self.get_datestr(predict_date_idx)
            t_hour = np.full((self.img_y, self.img_x), int(cur_datestr[8:10])/24).reshape(1, self.img_y, -1)
            t_day = np.full((self.img_y, self.img_x), int(cur_datestr[6:8])/31).reshape(1, self.img_y, -1)
            t_month = np.full((self.img_y, self.img_x), int(cur_datestr[4:6])/12).reshape(1, self.img_y, -1)
            # radar_history = torch.FloatTensor(np.concatenate((self.geo_lat, self.geo_lon, t_hour, t_day, t_month, radar_history), axis = 0))
            radar_history = torch.FloatTensor(np.concatenate((radar_history, t_hour, t_day, t_month, self.geo_lat, self.geo_lon), axis = 0))

        gt_list = []
        for i in range(6):
            gt, mask = parse(predict_date_idx+i, only_image = False)
            gt_cls = torch.zeros_like(gt)
            if self.cls_num == 3:
                gt_cls[gt >= 1.] = 1
                gt_cls[gt >= 10.] = 2
            elif self.cls_num == 8:
                gt_cls[gt >= 0.1] = 1
                gt_cls[gt >= 1.] = 2
                gt_cls[gt >= 5.] = 3
                gt_cls[gt >= 10.] = 4
                gt_cls[gt >= 20.] = 5
                gt_cls[gt >= 40.] = 6
                gt_cls[gt >= 70.] = 7
            else:
                print("Cls Num Error!")
            gt_list.append(gt_cls.unsqueeze(0))

        gt_cls = torch.cat(gt_list, axis=0)
        lead_cls = torch.full(gt.shape, (lead-1))
        # #(b, c, t, w , h)
        radar_history = radar_history.permute(1,0,2,3)

        return radar_history, gt_cls, mask, (lead-1), lead_cls
        
