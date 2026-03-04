import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import optuna, random, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import learn2learn as l2l
from torchvision.models.video import r3d_18, R3D_18_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50      
TASKS_PER_EPOCH = 30
VAL_TASKS = 15

CURRENT_OPTICAL_FLOW_PATH = ""
CURRENT_SPEAKERS = []
CURRENT_TOOL = "wav2lip" 

class OpticalFlowModel(nn.Module):
    def __init__(self, dropout, use_resnet=False):
        super().__init__()
        self.use_resnet = use_resnet
        if self.use_resnet:
            try: self.backbone = r3d_18(weights=R3D_18_Weights.DEFAULT)
            except: self.backbone = r3d_18(weights=None)
            old_layer = self.backbone.stem[0]
            self.backbone.stem[0] = nn.Conv3d(2, old_layer.out_channels, kernel_size=old_layer.kernel_size, stride=old_layer.stride, padding=old_layer.padding, bias=old_layer.bias is not None)
            self.backbone.fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 2))
        else:
            self.features = nn.Sequential(
                nn.Conv3d(2, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d((1, 2, 2)),
                nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(), nn.MaxPool3d((1, 2, 2)),
                nn.Conv3d(64, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(), nn.MaxPool3d((1, 2, 2)),
                nn.Conv3d(128, 256, 3, padding=1), nn.BatchNorm3d(256), nn.ReLU(), nn.MaxPool3d((1, 2, 2)),
                nn.AdaptiveAvgPool3d((1, 4, 4)) 
            )
            self.classifier = nn.Sequential(nn.Linear(256 * 4 * 4, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 2))

    def forward(self, x):
        if self.use_resnet: return self.backbone(x)
        else: return self.classifier(self.features(x).view(x.size(0), -1))

class FusionOpticalFlowModel(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.eye_branch = nn.Sequential(nn.Conv3d(2, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d((1, 2, 2)), nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(), nn.MaxPool3d((1, 2, 2)), nn.Conv3d(64, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(), nn.MaxPool3d((1, 2, 2)), nn.AdaptiveAvgPool3d((1, 1, 1))) 
        self.lip_branch = nn.Sequential(nn.Conv3d(2, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d((1, 2, 2)), nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(), nn.MaxPool3d((1, 2, 2)), nn.Conv3d(64, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(), nn.MaxPool3d((1, 2, 2)), nn.AdaptiveAvgPool3d((1, 1, 1))) 
        self.classifier = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 2))

    def forward(self, x):
        x_eyes = x[:, :, :, :32, :] 
        x_lips = x[:, :, :, 32:, :] 
        combined = torch.cat((self.eye_branch(x_eyes).view(x.size(0), -1), self.lip_branch(x_lips).view(x.size(0), -1)), dim=1)
        return self.classifier(combined)

class TaskGenerator:
    def __init__(self, base_path, speakers, shots, tool_filter='wav2lip'):
        self.base_path = base_path
        self.speakers = speakers
        self.shots = shots
        self.tool_filter = tool_filter
        self.max_frames = 30
    
    def _process_flow(self, flow_data):
        try:
            total = flow_data.shape[0]
            if total < self.max_frames:
                flow = torch.from_numpy(flow_data).float().permute(3, 0, 1, 2)
                flow = F.pad(flow, (0, 0, 0, 0, 0, self.max_frames - total))
            else:
                start = random.randint(0, total - self.max_frames)
                flow = torch.from_numpy(flow_data[start : start + self.max_frames]).float().permute(3, 0, 1, 2)
            
            min_v, max_v = flow.min(), flow.max()
            if max_v - min_v > 0: flow = (flow - min_v) / (max_v - min_v)
            return flow
        except: return None

    def _load_files_from_dir(self, path, lbl):
        if not os.path.exists(path): return []
        all_files = [f for f in os.listdir(path) if f.endswith('.npy')]
        filtered = []
        if lbl == 'fake':
            if self.tool_filter == 'synthetic': filtered = [f for f in all_files if 'synth_' in f]
            else:
                for f in all_files:
                    if self.tool_filter == 'wav2lip' and 'fake_w2l_' in f: filtered.append(f)
                    elif self.tool_filter == 'fomm' and 'fake_fomm_' in f: filtered.append(f)
                    elif self.tool_filter == 'liveportrait' and 'fake_lp_' in f: filtered.append(f)
                    elif self.tool_filter == 'both' or self.tool_filter == 'all': filtered.append(f)
        else: filtered = all_files 
        return sorted([os.path.join(path, f) for f in filtered])

    def load_n_samples(self, files, n_shots):
        tensors = []
        if not files: return []
        if len(files) >= n_shots:
            selected = random.sample(files, n_shots)
            for f in selected:
                try:
                    d = np.load(f, allow_pickle=True)
                    t = self._process_flow(d)
                    if t is not None: tensors.append(t)
                except: pass
        else:
            loaded_data = []
            for f in files:
                try: loaded_data.append(np.load(f, allow_pickle=True))
                except: pass
            if not loaded_data: return []
            for _ in range(n_shots):
                data = random.choice(loaded_data)
                t = self._process_flow(data) 
                if t is not None: tensors.append(t)
        return tensors[:n_shots]

    def create_task(self):
        if len(self.speakers) < 1: return None, None, None, None
        for _ in range(20): 
            s1 = random.choice(self.speakers)
            s2 = random.choice(self.speakers) 
            
            if self.tool_filter == 'synthetic':
                if 'optical_flow' in self.base_path:
                    fake_root = os.path.join(self.base_path.split('optical_flow')[0], 'optical_flow', 'combined', 'fake_synthetic')
                    real_root_s1 = os.path.join(self.base_path, 'real', s1)
                    real_root_s2 = os.path.join(self.base_path, 'real', s2)
                else:
                    fake_root = os.path.join(self.base_path, 'optical_flow', 'combined', 'fake_synthetic')
                    real_root_s1 = os.path.join(self.base_path, 'optical_flow', 'combined', 'real', s1)
                    real_root_s2 = os.path.join(self.base_path, 'optical_flow', 'combined', 'real', s2)
                
                sr_files = self._load_files_from_dir(real_root_s1, 'real')
                qr_files = self._load_files_from_dir(real_root_s2, 'real')
                all_fakes = self._load_files_from_dir(fake_root, 'fake')
                sf_files = all_fakes 
                qf_files = all_fakes
            else:
                sr_files = self._load_files_from_dir(os.path.join(self.base_path, 'real', s1), 'real')
                sf_files = self._load_files_from_dir(os.path.join(self.base_path, 'fake', s1), 'fake')
                qr_files = self._load_files_from_dir(os.path.join(self.base_path, 'real', s2), 'real')
                qf_files = self._load_files_from_dir(os.path.join(self.base_path, 'fake', s2), 'fake')

            s_real = self.load_n_samples(sr_files, self.shots)
            s_fake = self.load_n_samples(sf_files, self.shots)
            q_real = self.load_n_samples(qr_files, self.shots)
            q_fake = self.load_n_samples(qf_files, self.shots)
            
            if len(s_real) < self.shots or len(s_fake) < self.shots or len(q_real) < self.shots or len(q_fake) < self.shots: continue
            
            s_x = torch.stack(s_real + s_fake)
            s_y = torch.tensor([0]*len(s_real) + [1]*len(s_fake))
            q_x = torch.stack(q_real + q_fake)
            q_y = torch.tensor([0]*len(q_real) + [1]*len(q_fake))
            return s_x.to(device), s_y.to(device), q_x.to(device), q_y.to(device)
        return None, None, None, None

def objective(trial):
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    inner_lr = trial.suggest_float("inner_lr", 0.001, 0.03)
    meta_lr = trial.suggest_float("meta_lr", 1e-5, 1e-3)
    shots = trial.suggest_int("shots", 3, 10)

    if CURRENT_TOOL == 'synthetic': model = FusionOpticalFlowModel(dropout).to(device)
    else:
        is_ff = "FaceForensics" in CURRENT_OPTICAL_FLOW_PATH
        model = OpticalFlowModel(dropout, use_resnet=is_ff).to(device)

    maml = l2l.algorithms.MAML(model, lr=inner_lr)
    optimizer = optim.Adam(maml.parameters(), lr=meta_lr)
    spks = CURRENT_SPEAKERS
    split = int(len(spks) * 0.8)
    train_gen = TaskGenerator(CURRENT_OPTICAL_FLOW_PATH, spks[:split], shots, tool_filter=CURRENT_TOOL)
    val_gen = TaskGenerator(CURRENT_OPTICAL_FLOW_PATH, spks[split:], shots, tool_filter=CURRENT_TOOL)

    for _ in range(EPOCHS):
        for _ in range(TASKS_PER_EPOCH):
            s_x, s_y, q_x, q_y = train_gen.create_task()
            if s_x is None: continue
            learner = maml.clone()
            learner.adapt(F.cross_entropy(learner(s_x), s_y))
            optimizer.zero_grad()
            F.cross_entropy(learner(q_x), q_y).backward()
            optimizer.step()

    val_accs = []
    for _ in range(VAL_TASKS):
        s_x, s_y, q_x, q_y = val_gen.create_task()
        if s_x is None: continue
        learner = maml.clone()
        learner.adapt(F.cross_entropy(learner(s_x), s_y))
        out = learner(q_x)
        acc = (out.argmax(1) == q_y).float().mean().item()
        val_accs.append(acc)

    return np.mean(val_accs) if val_accs else 0.0

def run_hyperparameter_tuning(data_path, speakers, tool='wav2lip', n_trials=5):
    global CURRENT_OPTICAL_FLOW_PATH, CURRENT_SPEAKERS, CURRENT_TOOL
    CURRENT_OPTICAL_FLOW_PATH = data_path
    CURRENT_SPEAKERS = speakers
    CURRENT_TOOL = tool 
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_trial.params