import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random, numpy as np, torch, json
import torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from sklearn.metrics import (
    precision_recall_fscore_support, confusion_matrix, roc_curve, auc, 
    precision_recall_curve
)
from sklearn.calibration import calibration_curve 
from tqdm import tqdm
import learn2learn as l2l
import time
import matplotlib.pyplot as plt
import seaborn as sns
from src import tune_optuna 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OpticalFlowModel = tune_optuna.OpticalFlowModel
FusionOpticalFlowModel = tune_optuna.FusionOpticalFlowModel

class TaskGenerator:
    def __init__(self, base_path, speakers, shots, tool_filter='wav2lip'):
        self.base_path = base_path
        self.speakers = speakers
        self.shots = shots
        self.tool_filter = tool_filter
        self.max_frames = 30
    
    def _process_flow(self, flow_data):
        try:
            total_frames = flow_data.shape[0]
            
            if total_frames < self.max_frames:
                flow = torch.from_numpy(flow_data).float().permute(3, 0, 1, 2)
                flow = F.pad(flow, (0, 0, 0, 0, 0, self.max_frames - total_frames))
                
                min_val, max_val = flow.min(), flow.max()
                if max_val - min_val > 0: flow = (flow - min_val) / (max_val - min_val)
                return flow

            start = random.randint(0, total_frames - self.max_frames)
            segment = flow_data[start : start + self.max_frames]
            
            flow = torch.from_numpy(segment).float().permute(3, 0, 1, 2)
            min_val, max_val = flow.min(), flow.max()
            if max_val - min_val > 0: flow = (flow - min_val) / (max_val - min_val)
            return flow
        except: return None

    def _load_files_from_dir(self, path, lbl):
        if not os.path.exists(path): return []
        all_files = [f for f in os.listdir(path) if f.endswith('.npy')]
        filtered = []
        
        if lbl == 'fake':
            if self.tool_filter == 'synthetic':
                 filtered = [f for f in all_files if 'synth_' in f]
            else:
                for f in all_files:
                    if self.tool_filter == 'wav2lip' and 'fake_w2l_' in f: filtered.append(f)
                    elif self.tool_filter == 'fomm' and 'fake_fomm_' in f: filtered.append(f)
                    elif self.tool_filter == 'liveportrait' and 'fake_lp_' in f: filtered.append(f)
                    elif self.tool_filter == 'both' or self.tool_filter == 'all': filtered.append(f)
        else:
            filtered = all_files 
        
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
                try:
                    d = np.load(f, allow_pickle=True)
                    loaded_data.append(d)
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
            
            if len(s_real) < self.shots or len(s_fake) < self.shots or \
               len(q_real) < self.shots or len(q_fake) < self.shots:
                continue
            
            s_x = torch.stack(s_real + s_fake)
            s_y = torch.tensor([0]*len(s_real) + [1]*len(s_fake))
            q_x = torch.stack(q_real + q_fake)
            q_y = torch.tensor([0]*len(q_real) + [1]*len(q_fake))
            
            return s_x.to(device), s_y.to(device), q_x.to(device), q_y.to(device)

        return None, None, None, None

def generate_production_graphs(save_dir, run_name, history, test_data):
    print(f"    Generating 10 Production Graphs for {run_name}...")
    sns.set(style="whitegrid")
    
    y_true = test_data['labels']
    y_pred = test_data['preds']
    y_probs = test_data['probs']
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Meta-Train Loss', color='#E63946', linewidth=2)
    plt.title(f'1. Training Loss Convergence - {run_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{run_name}_01_loss.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(history['val_acc'], label='Accuracy', linewidth=2)
    plt.plot(history['val_f1'], label='F1 Score', linestyle='--', alpha=0.7)
    plt.title(f'2. Validation Metrics Over Time - {run_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{run_name}_02_val_metrics.png"))
    plt.close()

    if 'adapt_gain' in history and len(history['adapt_gain']) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(history['adapt_gain'], color='#2A9D8F', linewidth=2)
        plt.title(f'3. Meta-Learning Gain - {run_name}')
        plt.xlabel('Epochs')
        plt.ylabel('Gain (%)')
        plt.axhline(0, color='black', linestyle='--')
        plt.savefig(os.path.join(save_dir, f"{run_name}_03_adapt_gain.png"))
        plt.close()

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title(f'4. Final Confusion Matrix - {run_name}')
    plt.savefig(os.path.join(save_dir, f"{run_name}_04_confusion_matrix.png"))
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title(f'5. ROC Curve - {run_name}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, f"{run_name}_05_roc_curve.png"))
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, color='purple', lw=2)
    plt.title(f'6. Precision-Recall Curve - {run_name}')
    plt.savefig(os.path.join(save_dir, f"{run_name}_06_pr_curve.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(x=y_probs, hue=y_true, kde=True, bins=20, palette=['green', 'red'], element="step")
    plt.title(f'7. Confidence Distribution - {run_name}')
    plt.savefig(os.path.join(save_dir, f"{run_name}_07_confidence_dist.png"))
    plt.close()

    if cm.sum(axis=1).min() > 0:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        class_acc = cm_norm.diagonal()
        plt.figure(figsize=(8, 6))
        sns.barplot(x=['Real', 'Fake'], y=class_acc, palette=['#66C2A5', '#FC8D62'])
        plt.ylim(0, 1)
        plt.title(f'8. Class-wise Accuracy - {run_name}')
        plt.savefig(os.path.join(save_dir, f"{run_name}_08_class_acc.png"))
        plt.close()

    try:
        prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)
        plt.figure(figsize=(10, 6))
        plt.plot(prob_pred, prob_true, marker='o', label='Model')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
        plt.title(f'9. Calibration Curve - {run_name}')
        plt.savefig(os.path.join(save_dir, f"{run_name}_09_calibration.png"))
        plt.close()
    except: pass

    if 'task_losses' in history and history['task_losses']:
        plt.figure(figsize=(8, 6))
        sns.boxplot(y=history['task_losses'], color='lightblue')
        plt.title(f'10. Task Loss Stability - {run_name}')
        plt.savefig(os.path.join(save_dir, f"{run_name}_10_task_loss_dist.png"))
        plt.close()

def evaluate_model(maml, task_gen, num_tasks=30):
    accs, preds, labels, probs = [], [], [], []
    pre_adapt_accs = []
    
    for i in range(num_tasks):
        s_x, s_y, q_x, q_y = task_gen.create_task()
        if s_x is None: continue
            
        learner = maml.clone()
        with torch.no_grad():
            pre_out = learner(q_x)
            pre_pred = pre_out.argmax(1)
            pre_adapt_accs.append((pre_pred == q_y).float().mean().item())

        learner.adapt(F.cross_entropy(learner(s_x), s_y))
        
        with torch.no_grad():
            out = learner(q_x)
            prob = F.softmax(out, dim=1)[:, 1] 
            predicted_classes = out.argmax(1)
        
        correct = (predicted_classes == q_y).float().sum().item()
        total = q_y.size(0)
        accs.append(correct / total)
        preds.extend(predicted_classes.cpu().tolist())
        labels.extend(q_y.cpu().tolist())
        probs.extend(prob.cpu().tolist())

    if not accs: return None

    acc = np.mean(accs) * 100
    adapt_gain = acc - (np.mean(pre_adapt_accs) * 100)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    
    return {
        "Accuracy": acc, 
        "Precision": precision, 
        "Recall": recall, 
        "F1": f1,
        "Adapt_Gain": adapt_gain,
        "raw_labels": labels,
        "raw_preds": preds,
        "raw_probs": probs
    }

def run_training_pipeline(data_path, speakers, run_name="Default", tool="both", save_path=None, is_production=False):
    print(f"\n Pipeline: {run_name} | Tool: {tool} | Production: {is_production}")
    start_time = time.time()
    
    flat_speakers = [s.replace('\\', '/').split('/')[-1] for s in speakers]

    results_dir = os.path.dirname(save_path) if save_path else "results"
    os.makedirs(results_dir, exist_ok=True)
    
    params_file = os.path.join(results_dir, f"best_params_{run_name}.json")
    
    if os.path.exists(params_file):
        print(f"   Loading cached hyperparameters from {params_file}")
        with open(params_file, 'r') as f: best_params = json.load(f)
    else:
        print("   -> Tuning Hyperparameters...")
        tune_speakers = flat_speakers[:6] if len(flat_speakers) > 6 else flat_speakers
        best_params = tune_optuna.run_hyperparameter_tuning(data_path, tune_speakers, tool=tool, n_trials=5)
        if save_path:
            with open(params_file, 'w') as f: json.dump(best_params, f)

    dropout = best_params["dropout"]
    inner_lr = best_params["inner_lr"]
    meta_lr = best_params["meta_lr"]
    shots = int(best_params["shots"])

    total = len(flat_speakers)
    if total < 10:
        train_s = val_s = test_s = flat_speakers
    else:
        n_test = int(total * 0.10)
        n_val = int(total * 0.10)
        if n_test < 2: n_test = 2
        if n_val < 2: n_val = 2
        test_s = flat_speakers[:n_test]
        val_s = flat_speakers[n_test : n_test + n_val]
        train_s = flat_speakers[n_test + n_val:]

    if tool == 'synthetic':
        print("   [Config] Using Fusion Architecture (Two-Stream) for Synthetic Data")
        model = FusionOpticalFlowModel(dropout).to(device)
    else:
        print("   [Config] Using Standard 3D-CNN (Single Stream)")
        is_ff = "FaceForensics" in data_path or "faceforensics" in data_path.lower()
        model = OpticalFlowModel(dropout, use_resnet=is_ff).to(device)

    maml = l2l.algorithms.MAML(model, lr=inner_lr)
    
    history = {'train_loss': [], 'val_acc': [], 'val_f1': [], 'adapt_gain': [], 'task_losses': []}
    optimizer = optim.Adam(maml.parameters(), lr=meta_lr)
    
    train_gen = TaskGenerator(data_path, train_s, shots, tool_filter=tool)
    val_gen = TaskGenerator(data_path, val_s, shots, tool_filter=tool) 

    epochs = 50 if is_production else 5
    
    print(f"   -> Training MAML ({epochs} Epochs)...")
    for epoch in tqdm(range(epochs), desc="   Epochs"):
        epoch_loss = 0.0
        tasks_per_epoch = 30
        epoch_task_losses = [] 
        
        for _ in range(tasks_per_epoch):
            s_x, s_y, q_x, q_y = train_gen.create_task()
            if s_x is None: continue
            learner = maml.clone()
            learner.adapt(F.cross_entropy(learner(s_x), s_y))
            loss = F.cross_entropy(learner(q_x), q_y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            
            val = loss.item()
            epoch_loss += val
            if epoch == epochs - 1: epoch_task_losses.append(val)
        
        avg_loss = epoch_loss / tasks_per_epoch if tasks_per_epoch > 0 else 0
        history['train_loss'].append(avg_loss)
        if epoch_task_losses: history['task_losses'] = epoch_task_losses

        val_interval = 2 if is_production else 5
        if (epoch + 1) % val_interval == 0 or epoch == epochs - 1:
            val_metrics = evaluate_model(maml, val_gen, num_tasks=15)
            if val_metrics:
                history['val_acc'].append(val_metrics['Accuracy'])
                history['val_f1'].append(val_metrics['F1'])
                history['adapt_gain'].append(val_metrics['Adapt_Gain'])

    if save_path:
        torch.save({'model_state_dict': model.state_dict(), 'hyperparams': best_params}, save_path)

    print("   -> Evaluating on Test Set...")
    test_gen = TaskGenerator(data_path, test_s, shots, tool_filter=tool)
    metrics = evaluate_model(maml, test_gen, num_tasks=50)
    
    if metrics:
        end_time = time.time()
        metrics["Time (s)"] = round(end_time - start_time, 2)
        if is_production:
            test_data = {
                'labels': metrics['raw_labels'],
                'preds': metrics['raw_preds'],
                'probs': metrics['raw_probs']
            }
            generate_production_graphs(results_dir, run_name, history, test_data)
        del metrics['raw_labels'], metrics['raw_preds'], metrics['raw_probs']
    else:
        print("    Evaluation Failed.")
        metrics = {"Accuracy": 0, "Time (s)": 0}
    
    return metrics, best_params, save_path