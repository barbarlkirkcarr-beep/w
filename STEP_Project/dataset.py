import torch
from torch.utils.data import Dataset
from config import Config

class PedestrianDataset(Dataset):
    def __init__(self, mode='train'):
        self.mode = mode
        # 实际项目中，这里读取 txt 文件并解析
        # self.data = load_eth_ucy_data(...)
        self.num_samples = 100 # 模拟数据量

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 模拟：观测轨迹
        obs_traj = torch.randn(Config.OBS_LEN, 2)
        # 模拟：未来真值轨迹
        pred_gt = torch.randn(Config.PRED_LEN, 2)
        
        # 模拟：场景输入
        # 通道 = 语义分割类别 + 历史轨迹热力图
        c = Config.NUM_SEMANTIC_CLASSES + 1
        scene_input = torch.zeros(c, Config.IMG_SIZE[0], Config.IMG_SIZE[1])
        # 随机激活一些像素模拟语义分割
        scene_input[:Config.NUM_SEMANTIC_CLASSES, 100:150, 100:150] = 1.0
        
        # 模拟：终点真值概率图 (Gaussian Heatmap)
        goal_map_gt = torch.zeros(1, Config.IMG_SIZE[0], Config.IMG_SIZE[1])
        goal_map_gt[0, 128, 128] = 1.0 # 假设终点在中心
        
        return obs_traj, pred_gt, scene_input, goal_map_gt