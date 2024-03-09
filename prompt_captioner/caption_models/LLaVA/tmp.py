
from PIL import Image
from torchvision import transforms
import os.path
from argparse import ArgumentParser
from torch.utils import data
import json
import torch
import torch.distributed as dist
import os
import os.path as osp
from os.path import join as  opj
import pandas as pd
from random import randint
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import decord
import glob
import subprocess
import time



class my_dataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.shuffle = True
        self.resolution = args.resolution

        if args.train_file.endswith('.csv'):
            self.train_file = pd.read_csv(args.train_file)
        elif args.train_file.endswith('.json'):
            # coco_vat_vat0_11_all_id_rootfolder_clsidx_spacy.json 
            # 格式：   id :  {   'idx_list' :  [0],  'root_folder'  :  'coco_vat_9' }

            if hasattr(args, 'part_nums') and args.part_nums >1:
                self.part_nums = args.part_nums
            else:
                self.part_nums = 100000
            self.part_index = args.part_index
            t1 = time.time()
            with open(args.train_file, 'r', encoding='utf-8') as f:
                self.train_file = json.load(f)
                if type(self.train_file) is str:
                    self.train_file = json.loads(self.train_file)

                self.id_list = list(self.train_file.keys())
                #=============================
                # obtain subset of self.id_list, so that deduplication time is less than 30min 
                self.id_list = self.id_list[self.part_nums*(self.part_index-1):self.part_nums*self.part_index]


                print(f'Nums of train_file is {len(self.id_list)},part_index:{self.part_index}, first:{self.id_list[0]}')
                self.no_caption_id_list = []
                self.exist_caption_path_list = {}
                for idx, id in enumerate(self.id_list):
                    caption_json = osp.join('/apdcephfs_cq3/share_1311970/A_Youtube',self.train_file[id]['root_folder'],f'{id}_caption.json')
                    if not os.path.exists(caption_json):
                        mp4_path = osp.join('/apdcephfs_cq3/share_1311970/A_Youtube',self.train_file[id]['root_folder'],f'{id}.mp4')
                        self.no_caption_id_list.append(mp4_path)
                    else:
                        self.exist_caption_path_list[caption_json]=True

                    if idx%10000==0:
                        print(f'Time_cost:{time.time()-t1}s, idx:{idx}, caption_json:{caption_json}')
                print(f'Nums of no_caption_id_list is {len(self.no_caption_id_list)}, first:{self.no_caption_id_list[0]}')
                print(f'Nums of exist_caption_path_list is {len(self.exist_caption_path_list)}')
            t2 = time.time()
            print(f'Time cost:{t2-t1}s')
        # print('======',self.exist_file_list,'====',self.no_caption_id_list)

        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        self.patch_resize_transform = transforms.Compose([
                        lambda image: image.convert("RGB"),
                        transforms.Resize((self.resolution, self.resolution), interpolation=Image.BICUBIC),
                        transforms.ToTensor(), 
                        transforms.Normalize(mean=mean, std=std)
                    ])
        print('Dataset nums is {}'.format(self.__len__()))
        time.sleep(10)


    def __len__(self):
        return len(self.no_caption_id_list)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, idx):
        if idx >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(idx + 1)

    def skip_sample(self, idx):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(idx=idx)


    def get_frames_from_video_opencv(self, batchsize=1, video_path=None, caption_nums_per_video=8):
        # 加载视频
        video_path = video_path
        cap = cv2.VideoCapture(video_path)

        # 确定要提取的帧数
        num_frames = caption_nums_per_video
        # 计算每隔多少帧提取一次
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = total_frames // num_frames

        # 用于存储提取的图像的tensor
        frames = torch.empty(num_frames, 3, self.resolution, self.resolution)

        # 直接读取指定帧
        for i in range(num_frames):
            # 计算要提取的帧的索引
            idx = i * step
            # 设置当前帧为所需的帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            # 读取该帧
            ret, frame = cap.read()
            if not ret:
                break

            # 转换为PIL Image并进行缩放
            Image_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame = self.patch_resize_transform(Image_frame)

            # 将numpy数组转换为tensor并存储在frames中
            frames[i] = frame

        # 打印输出frames的形状
        # print(frames.shape)
        cap.release()
        return frames


    def get_frames_from_video(self, batchsize=1, video_path=None, caption_nums_per_video = 8, ):

        # 加载视频
        video_path = video_path
        vr = decord.VideoReader(video_path)
        # 确定要提取的帧数
        num_frames = caption_nums_per_video
        # 计算每隔多少帧提取一次
        step = len(vr) // num_frames
        # 用于存储提取的图像的tensor
        frames = torch.empty(num_frames, 3, self.resolution, self.resolution)

        # 从视频中提取图像
        for i in range(num_frames):
            # 计算要提取的帧的索引
            idx = i * step
            # 从视频中读取帧
            decord_frame = vr[idx].asnumpy()
            Image_frame = Image.fromarray(decord_frame)
            frame = self.patch_resize_transform(Image_frame)#.unsqueeze(0)
            
            # 将numpy数组转换为tensor并存储在frames中
            frames[i] = frame

        # 打印输出frames的形状
        # print(frames.shape)
        vr.close()
        return frames


    def __getitem__(self, idx):

        try:
        
            # video_id = self.filter_train_file[idx]
            # video_path = opj(self.vat_root, video_id)
            video_path = self.no_caption_id_list[idx]
            video_id = video_path.split('/')[-1].split('.')[0]
            # 假如多个程序一起跑，其他已经生成了，就跳过
            caption_video_json =  video_path.replace('.mp4', '_caption.json')
            if  caption_video_json in self.exist_caption_path_list:
                print('parallel task has process it :{}'.format(caption_video_json))
                # return '===========', None, torch.random(8,3,self.resolution, self.resolution)
                return self.skip_sample(idx)
            # if os.path.exists(caption_video_json):
            #     print('parallel task has process it :{}'.format(caption_video_json))
            #     # return '===========', None, torch.random(8,3,self.resolution, self.resolution)
            #     return self.skip_sample(idx)
            if not osp.exists(video_path):
                print('video {} is not exists and skip this idx! '.format(video_path))
                return self.skip_sample(idx)
            video_frames = self.get_frames_from_video_opencv( video_path = video_path, caption_nums_per_video = args.caption_nums_per_video)
            
            return video_id, video_path, video_frames
            
        except Exception as e:
            print('Read video error in {},{} and we have skip this !, this will not cause error!'.format(idx,e))
            return self.skip_sample(idx)

        
class my_dataset2(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.shuffle = True
        self.resolution = args.resolution

        if args.train_file.endswith('.csv'):
            self.train_file = pd.read_csv(args.train_file)
        elif args.train_file.endswith('.json'):
            # coco_vat_vat0_11_all_id_rootfolder_clsidx_spacy.json 
            # 格式：   id :  {   'idx_list' :  [0],  'root_folder'  :  'coco_vat_9' }

            if hasattr(args, 'part_nums') and args.part_nums >1:
                self.part_nums = args.part_nums
            else:
                self.part_nums = 100000
            self.part_index = args.part_index
            t1 = time.time()
            with open(args.train_file, 'r', encoding='utf-8') as f:
                self.train_file = json.load(f)
                if type(self.train_file) is str:
                    self.train_file = json.loads(self.train_file)

                self.id_list = list(self.train_file.keys())
                #=============================
                # obtain subset of self.id_list
                self.id_list = self.id_list[self.part_nums*(self.part_index-1):self.part_nums*self.part_index]


                print(f'Nums of train_file is {len(self.id_list)},part_index:{self.part_index}, first:{self.id_list[0]}')
                self.no_caption_id_list = []
                for idx,id in enumerate(self.id_list):
                    caption_json = osp.join('/apdcephfs_cq3/share_1311970/A_Youtube',self.train_file[id]['root_folder'],f'{id}_caption.json')
                    if not os.path.exists(caption_json):
                        mp4_path = osp.join('/apdcephfs_cq3/share_1311970/A_Youtube',self.train_file[id]['root_folder'],f'{id}.mp4')
                        self.no_caption_id_list.append(mp4_path)
                    if idx%10000==0:
                        print(f'Time_cost:{time.time()-t1}s, idx:{idx}, caption_json:{caption_json}')
                print(f'Nums of no_caption_id_list is {len(self.no_caption_id_list)}, first:{self.no_caption_id_list[0]}')
            t2 = time.time()
            print(f'Time cost:{t2-t1}s')
        # print('======',self.exist_file_list,'====',self.no_caption_id_list)

        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        self.patch_resize_transform = transforms.Compose([
                        lambda image: image.convert("RGB"),
                        transforms.Resize((self.resolution, self.resolution), interpolation=Image.BICUBIC),
                        transforms.ToTensor(), 
                        transforms.Normalize(mean=mean, std=std)
                    ])
        print('Dataset nums is {}'.format(self.__len__()))
        time.sleep(10)


    def __len__(self):
        return len(self.no_caption_id_list)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, idx):
        if idx >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(idx + 1)

    def skip_sample(self, idx):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(idx=idx)


    def get_frames_from_video_opencv(self, batchsize=1, video_path=None, caption_nums_per_video=8):
        # 加载视频
        video_path = video_path
        cap = cv2.VideoCapture(video_path)

        # 确定要提取的帧数
        num_frames = caption_nums_per_video
        # 计算每隔多少帧提取一次
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = total_frames // num_frames

        # 用于存储提取的图像的tensor
        frames = torch.empty(num_frames, 3, self.resolution, self.resolution)

        # 直接读取指定帧
        for i in range(num_frames):
            # 计算要提取的帧的索引
            idx = i * step
            # 设置当前帧为所需的帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            # 读取该帧
            ret, frame = cap.read()
            if not ret:
                break

            # 转换为PIL Image并进行缩放
            Image_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame = self.patch_resize_transform(Image_frame)

            # 将numpy数组转换为tensor并存储在frames中
            frames[i] = frame

        # 打印输出frames的形状
        # print(frames.shape)
        cap.release()
        return frames


    def get_frames_from_video(self, batchsize=1, video_path=None, caption_nums_per_video = 8, ):

        # 加载视频
        video_path = video_path
        vr = decord.VideoReader(video_path)
        # 确定要提取的帧数
        num_frames = caption_nums_per_video
        # 计算每隔多少帧提取一次
        step = len(vr) // num_frames
        # 用于存储提取的图像的tensor
        frames = torch.empty(num_frames, 3, self.resolution, self.resolution)

        # 从视频中提取图像
        for i in range(num_frames):
            # 计算要提取的帧的索引
            idx = i * step
            # 从视频中读取帧
            decord_frame = vr[idx].asnumpy()
            Image_frame = Image.fromarray(decord_frame)
            frame = self.patch_resize_transform(Image_frame)#.unsqueeze(0)
            
            # 将numpy数组转换为tensor并存储在frames中
            frames[i] = frame

        # 打印输出frames的形状
        # print(frames.shape)
        vr.close()
        return frames


    def __getitem__(self, idx):
        try:
        
            # video_id = self.filter_train_file[idx]
            # video_path = opj(self.vat_root, video_id)
            video_path = self.no_caption_id_list[idx]
            video_id = video_path.split('/')[-1].split('.')[0]
            # 假如多个程序一起跑，其他已经生成了，就跳过
            caption_video_json =  video_path.replace('.mp4', '_caption.json')
            if os.path.exists(caption_video_json):
                print('parallel task has process it :{}'.format(caption_video_json))
                # return '===========', None, torch.random(8,3,self.resolution, self.resolution)
                return self.skip_sample(idx)
            if not osp.exists(video_path):
                print('video {} is not exists and skip this idx! '.format(video_path))
                return self.skip_sample(idx)
            video_frames = self.get_frames_from_video_opencv( video_path = video_path, caption_nums_per_video = args.caption_nums_per_video)
            
            return video_id, video_path, video_frames
            
        except Exception as e:
            print('Read video error in {},{} and we have skip this !, this will not cause error!'.format(idx,e))
            return self.skip_sample(idx)
        

def test_dataset(args):

    train_dataset = my_dataset(args)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    from time import time
    for i, sample in enumerate(loader):
        video_ids, video_paths,  videos_frames = sample
        print(i, video_ids, video_paths, videos_frames.shape)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--caption_nums_per_video', type=int, default=8, help='process rank')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--train_file', type=str, default='/apdcephfs_cq3/share_1311970/A_Youtube/coco_vat_vat0_11_all_id_rootfolder_clsidx_spacy.json')
    # parser.add_argument('--vat_root', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--resolution', type=int, default=480)
    # parser.add_argument('--exist_caption_id_list_json', type=str, default=f'coco_vat_exist_caption_id_list_{month}{day}{hour}.json',help='')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')  # --dist_on_itp   ddp
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, help='url used to set up distributed training')
    parser.add_argument('--gpus', default=[0, 1, 2, 3], help='DP CUDA devices')
    parser.add_argument('--part_index', default=1, type=int, help='used to split train_file_id into different parts, and generate caption from part_index 1 to ....')
    parser.add_argument('--part_nums', default=1000, type=int, help='used to split train_file_id into different parts, and generate caption from part_index 1 to ....')
    args = parser.parse_args()
    test_dataset(args)

    """
    
    source /apdcephfs_cq3/share_1311970/lb/miniconda3/etc/profile.d/conda.sh
    conda activate /apdcephfs_cq3/share_1311970/lb/miniconda3/envs/pytorch1.12.1
    cd /apdcephfs_cq3/share_1311970/A_ofa
    python3  ofa_dataset.py \
    --train_file  "/apdcephfs_cq3/share_1311970/A_Youtube/coco_vat_890w_id_title_folderidx_merge.json"   \
    --num_workers  10   --batch_size 3    \
    --part_index 9 \
    --part_nums  100000
    
    
    """