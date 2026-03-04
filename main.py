import os
import pandas as pd
from datetime import datetime
from src.generate_fakes import generate_fake_videos
from src.generate_synthetic import generate_synthetic_dataset
from src.extract_optical_flow import process_dataset_mode
from src.train_maml import run_training_pipeline
import sys

try:
    import openpyxl
except ImportError:
    print("Warning: 'openpyxl' library not found. Excel logging might fail. Install via: pip install openpyxl")

BACKEND_ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT_DATA_DIR = os.path.join(BACKEND_ROOT, "data")
WAV2LIP_DIR = os.path.join(BACKEND_ROOT, "Wav2Lip")
FOMM_DIR = os.path.join(BACKEND_ROOT, "first-order-model")
LP_DIR = os.path.join(BACKEND_ROOT, "LivePortrait")
MODEL_SAVE_DIR = os.path.join(BACKEND_ROOT, "saved_models")
RESULTS_DIR = os.path.join(BACKEND_ROOT, "results")

def get_speakers(base_dir, dataset_name):
    speakers = []
    if dataset_name == 'mobio':
        for source in ['idiap', 'unis']:
            path = os.path.join(base_dir, source)
            if os.path.exists(path):
                spks = [f"{source}/{s}" for s in os.listdir(path) if os.path.isdir(os.path.join(path, s))]
                speakers.extend(spks)
    elif dataset_name == 'faceforensics':
        vid_root = os.path.join(base_dir, "original_sequences", "youtube", "c23", "videos")
        if not os.path.exists(vid_root):
             vid_root = os.path.join(base_dir, "original_sequences")
        
        if os.path.exists(vid_root):
            speakers = sorted([os.path.splitext(f)[0] for f in os.listdir(vid_root) if f.endswith('.mp4')])
    else:
        vid_root = os.path.join(base_dir, "video")
        if os.path.exists(vid_root):
            speakers = sorted([d for d in os.listdir(vid_root) if d.startswith('s') and os.path.isdir(os.path.join(vid_root, d))])
    return sorted(speakers)

def log_results_to_excel(results_list, log_path):
    if not results_list: return
    df = pd.DataFrame(results_list)
    cols_order = [
        'Timestamp', 'Dataset', 'Region', 'Mode', 'Fake_Tool', 
        'Accuracy', 'F1', 'Precision', 'Recall', 'Shots', 'Time (s)', 'Model_Path'
    ]
    cols = [c for c in cols_order if c in df.columns]
    df = df[cols]
    
    if not os.path.exists(log_path):
        df.to_excel(log_path, index=False, sheet_name='Logs')
    else:
        with pd.ExcelWriter(log_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            if 'Logs' in writer.book.sheetnames:
                start_row = writer.sheets['Logs'].max_row
                df.to_excel(writer, index=False, header=False, startrow=start_row, sheet_name='Logs')
            else:
                df.to_excel(writer, index=False, sheet_name='Logs')

def run_pipeline_for_dataset(dataset_name, mode_choice, reg_choice, current_tool_variant, root_dir, wav2lip_dir, fomm_dir, lp_dir, model_save_dir):
    
    if dataset_name == 'grid':
        base_dir = os.path.join(root_dir, "gridcorpus")
    elif dataset_name == 'mobio':
        base_dir = os.path.join(root_dir, "mobio")
    elif dataset_name == 'faceforensics':
        base_dir = os.path.join(root_dir, "FaceForensics")
    else: return []

    all_speakers = get_speakers(base_dir, dataset_name)
    if not all_speakers: 
        print(f" [Error] No speakers found for {dataset_name} at {base_dir}")
        return []

    if mode_choice == '1':
        limit = 5
        target_speakers = all_speakers[:limit]
        video_limit = 500
        mode_suffix = "test"
        is_production = False
        mode_str = "Test"
    else:
        target_speakers = all_speakers
        video_limit = 500
        mode_suffix = "prod"
        is_production = True
        mode_str = "Production"

    if current_tool_variant == 'synthetic':
        print(f"\n[Step 1] Generating Synthetic Incoherence Fakes for {dataset_name}...")
        generate_synthetic_dataset(base_dir, dataset_name, videos_limit=video_limit)
        modes = ['combined']
    else:
        print(f"\n[Step 1] Checking/Generating Fakes ({current_tool_variant})...")
        generate_fake_videos(
            base_dir, wav2lip_dir, fomm_dir, lp_dir, dataset_name, 
            video_limit, target_speakers, current_tool_variant
        )
        if reg_choice == '1': modes = ['lip']
        elif reg_choice == '2': modes = ['eye']
        elif reg_choice == '3': modes = ['combined']
        elif reg_choice == '4': modes = ['lip', 'eye', 'combined']
        else: modes = ['combined']

    dataset_results = []
    
    for mode in modes:
        print(f"\n{'='*30}\n PROCESSING: {mode.upper()} | {current_tool_variant.upper()}\n{'='*30}")
        
        flow_dir = os.path.join(base_dir, "optical_flow", mode)
        output_dir_real = os.path.join(flow_dir, "real")
        
        if dataset_name == 'grid':
            input_dir_real = os.path.join(base_dir, "video")
        elif dataset_name == 'faceforensics':
            input_dir_real = os.path.join(base_dir, "original_sequences", "youtube", "c23", "videos")
            if not os.path.exists(input_dir_real): input_dir_real = base_dir
        else:
            input_dir_real = base_dir
        
        print(f"\n[Extraction] Ensuring REAL flow exists...")
        process_dataset_mode(
            input_dir_real, output_dir_real, mode, dataset_name,
            target_speakers, video_limit, is_fake=False
        )

        if current_tool_variant != 'synthetic':
            output_dir_fake = os.path.join(flow_dir, "fake")
            if current_tool_variant in ['wav2lip', 'both']:
                 process_dataset_mode(os.path.join(base_dir, "fake_dataset_w2l"), output_dir_fake, mode, dataset_name, target_speakers, video_limit, True)
            if current_tool_variant in ['fomm', 'both']:
                 process_dataset_mode(os.path.join(base_dir, "fake_dataset_fomm"), output_dir_fake, mode, dataset_name, target_speakers, video_limit, True)
            if current_tool_variant in ['liveportrait', 'both']:
                 process_dataset_mode(os.path.join(base_dir, "fake_dataset_lp"), output_dir_fake, mode, dataset_name, target_speakers, video_limit, True)

        tool_label = current_tool_variant
        save_file = os.path.join(model_save_dir, f"{dataset_name}_model_{mode}_{tool_label}_{mode_suffix}.pth")
        run_name = f"{dataset_name}_{mode}_{tool_label}_{mode_suffix}"
        
        metrics, best_params, model_path = run_training_pipeline(
            base_dir, 
            target_speakers, 
            run_name=run_name, 
            tool=current_tool_variant, 
            save_path=save_file,
            is_production=is_production 
        )
        
        if metrics:
            metrics['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            metrics['Dataset'] = dataset_name.upper()
            metrics['Region'] = mode.upper()
            metrics['Mode'] = mode_str
            metrics['Fake_Tool'] = tool_label.upper()
            metrics['Model_Path'] = os.path.basename(model_path)
            dataset_results.append(metrics)

    return dataset_results

def main():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("="*50 + "\n   VSA PIPELINE: SYNTHETIC PIVOT\n" + "="*50)
    print("Select Dataset:\n1. GRID Corpus\n2. MOBIO\n3. BOTH (Sequential)\n4. FaceForensics++ (New)")
    ds_choice = input("Choice: ").strip()
    
    print("\n1. Test Mode\n2. Production Mode")
    mode_choice = input("Choice: ").strip()
    
    print("\nSelect Method:\n1. Wav2Lip (Legacy)\n2. FOMM (Legacy)\n3. BOTH (Legacy)\n5. SYNTHETIC INCOHERENCE (New/Recommended)")
    tool_input = input("Choice: ").strip()

    datasets = []
    if ds_choice == '1': datasets = ['grid']
    elif ds_choice == '2': datasets = ['mobio']
    elif ds_choice == '3': datasets = ['grid', 'mobio']
    elif ds_choice == '4': datasets = ['faceforensics']
    
    tools_to_run = []
    if tool_input == '1': tools_to_run = ['wav2lip']
    elif tool_input == '2': tools_to_run = ['fomm']
    elif tool_input == '3': tools_to_run = ['wav2lip', 'fomm']
    elif tool_input == '5': tools_to_run = ['synthetic']
    else: tools_to_run = ['synthetic']
    
    reg_choice = '3' 
    if tool_input != '5':
        print("\nSelect Region:\n1. Lip\n2. Eye\n3. Combined\n4. All")
        reg_choice = input("Choice: ").strip()

    full_results = []
    for ds in datasets:
        for tool_variant in tools_to_run:
            print(f"\n>>> STARTING BATCH: {ds.upper()} + {tool_variant.upper()}")
            res = run_pipeline_for_dataset(ds, mode_choice, reg_choice, tool_variant, ROOT_DATA_DIR, WAV2LIP_DIR, FOMM_DIR, LP_DIR, MODEL_SAVE_DIR)
            full_results.extend(res)

    if full_results:
        df = pd.DataFrame(full_results)
        print("\n" + "="*50)
        print("FINAL RESULTS SUMMARY")
        print("="*50)
        display_cols = ['Dataset', 'Region', 'Fake_Tool', 'Accuracy', 'F1', 'Time (s)']
        available = [c for c in display_cols if c in df.columns]
        print(df[available].to_string(index=False))
        
        log_path = os.path.join(RESULTS_DIR, "pipeline_logs.xlsx")
        log_results_to_excel(full_results, log_path)
        print(f"\n[Log] Results appended to {log_path}")

if __name__ == "__main__":
    main()