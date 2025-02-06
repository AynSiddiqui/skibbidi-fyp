import argparse
import os
import shutil

def parse_args():
    default_train_dir = 'tests_may22'
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, required=False,
                        default=default_train_dir, help="training dir")
    return parser.parse_args()

def main():
    args = parse_args()
    train_dir = args.base_dir
    tokens = train_dir.split('\\')
    train_folder = tokens[-1].split('_')
    train_folder[0] = 'eval'
    tokens[-1] = '_'.join(train_folder)
    eval_dir = '\\'.join(tokens)
    eval_dir = os.path.normpath(eval_dir)

    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)
    folders = os.listdir(train_dir)
    
    for scenario in ['small_grid', 'large_grid']:
        scenario_dir = os.path.normpath(f"{eval_dir}\\{scenario}")
        
        if not os.path.exists(scenario_dir):
            os.mkdir(scenario_dir)
        
        train_data_dir = os.path.normpath(f"{scenario_dir}\\train_data")
        if not os.path.exists(train_data_dir):
            os.mkdir(train_data_dir)
        
        for coop in ['neighbor', 'global', 'local']:
            case = f"{scenario}_{coop}"
            if case not in folders:
                continue
            
            cur_folder = os.path.normpath(f"{train_dir}\\{case}")
            
            # Copy config file
            config_src = os.path.normpath(f"{cur_folder}\\data")
            config_dst = os.path.normpath(f"{cur_folder}\\model")
            for file in os.listdir(config_src):
                if file.endswith('.ini'):
                    shutil.copy(os.path.join(config_src, file), config_dst)
            
            # Copy model
            model_src = os.path.normpath(f"{cur_folder}\\model")
            model_dst = os.path.normpath(f"{scenario_dir}\\{coop}")
            if not os.path.exists(model_dst):
                shutil.copytree(model_src, model_dst)
            
            # Copy train reward
            tar_train_file = os.path.normpath(f"{train_data_dir}\\{coop}_train_reward.csv")
            train_reward_src = os.path.normpath(f"{cur_folder}\\log\\train_reward.csv")
            if os.path.exists(train_reward_src):
                shutil.copy(train_reward_src, tar_train_file)
        
        new_folder = os.path.normpath(f"{scenario_dir}\\naive")
        old_folder = os.path.normpath(f"{scenario_dir}\\local")
        os.mkdir(new_folder)
        
        # Copy .ini files from old_folder to new_folder
        for file in os.listdir(old_folder):
            if file.endswith('.ini'):
                shutil.copy(os.path.join(old_folder, file), new_folder)

if __name__ == '__main__':
    main()
