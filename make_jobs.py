import itertools
import os
import json
import argparse

# 各モデルが使用するパラメータの定義
MODEL_PARAM_MAP = {
    "LRGCN": ["label_max_hops", "label_temperature"],
    "Goodie": ["lp_alpha", "goodie_lamb"],
    "FISF": ["fisf_num_iterations", "fisf_alpha", "fisf_beta", "fisf_gamma", "fisf_mask_type"],
}

# デフォルト値の定義
DEFAULT_VALUES = {
    "label_max_hops": 1,
    "label_temperature": 0.25,
    "lp_alpha": 0.99,
    "goodie_lamb": 1,
    "fisf_num_iterations": 50,
    "fisf_alpha": 0.9,
    "fisf_beta": 0.7,
    "fisf_gamma": 0.1,
    "fisf_mask_type": "uniform",
}


# JSONファイルから設定を読み込む
def load_config(config_path="hyperparameters.json"):
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def get_model_params(config, model_name):
    """Get model-specific parameters from config."""
    model_specific = config.get("model_specific", {})
    return model_specific.get(model_name, {})


def get_params_for_model(config, model_name):
    """Get all parameters (with defaults) for a specific model."""
    global_config = config["global"]
    model_params = get_model_params(config, model_name)
    model_param_names = MODEL_PARAM_MAP.get(model_name, [])
    
    params = {}
    
    # グローバルパラメータ
    params["label_noise_rates"] = global_config["label_noise_rates"]
    params["feature_missing_rates"] = global_config["feature_missing_rates"]
    params["feature_noise_rates"] = global_config["feature_noise_rates"]
    params["hidden_channels"] = global_config["hidden_channels"]
    params["num_layers"] = global_config["num_layers"]
    
    # モデル固有パラメータ（モデルが使用するもののみ）
    for param_name in model_param_names:
        if param_name in model_params:
            params[param_name] = model_params[param_name]
        else:
            # デフォルト値を使用（リストとして）
            params[param_name] = [DEFAULT_VALUES.get(param_name, 1)]
    
    return params, model_param_names


def calculate_total_jobs(config):
    """Calculate total number of jobs that will be generated."""
    global_config = config["global"]
    datasets = global_config["datasets"]
    models = global_config["models"]
    
    total = 0
    for dataset_name, model in itertools.product(datasets, models):
        params, _ = get_params_for_model(config, model)
        jobs_per_combo = 1
        for param_values in params.values():
            jobs_per_combo *= len(param_values)
        total += jobs_per_combo
    return total


def build_command(dataset_name, model, params_dict, model_param_names, n_runs_value, epochs, early_stopping, patience, results_csv):
    """Build command string for a specific parameter combination."""
    cmd_parts = [
        f"python main.py --dataset_name {dataset_name} --model {model}",
        f"--label_noise_rate {params_dict['label_noise_rates']}",
        f"--feature_missing_rate {params_dict['feature_missing_rates']}",
        f"--feature_noise_rate {params_dict['feature_noise_rates']}",
        f"--hidden_channels {params_dict['hidden_channels']}",
        f"--num_layers {params_dict['num_layers']}",
    ]
    
    # モデル固有のパラメータのみ追加
    for param_name in model_param_names:
        if param_name in params_dict:
            if param_name == "label_max_hops":
                cmd_parts.append(f"--label_max_hops {params_dict[param_name]}")
            elif param_name == "label_temperature":
                cmd_parts.append(f"--label_temperature {params_dict[param_name]}")
            elif param_name == "lp_alpha":
                cmd_parts.append(f"--lp_alpha {params_dict[param_name]}")
            elif param_name == "goodie_lamb":
                cmd_parts.append(f"--goodie_lamb {params_dict[param_name]}")
            elif param_name == "fisf_num_iterations":
                cmd_parts.append(f"--fisf_num_iterations {params_dict[param_name]}")
            elif param_name == "fisf_alpha":
                cmd_parts.append(f"--fisf_alpha {params_dict[param_name]}")
            elif param_name == "fisf_beta":
                cmd_parts.append(f"--fisf_beta {params_dict[param_name]}")
            elif param_name == "fisf_gamma":
                cmd_parts.append(f"--fisf_gamma {params_dict[param_name]}")
            elif param_name == "fisf_mask_type":
                cmd_parts.append(f"--fisf_mask_type {params_dict[param_name]}")
    
    cmd_parts.extend([
        f"--n_runs {n_runs_value}",
        f"--epochs {epochs}",
        f"--early_stopping {early_stopping}",
        f"--patience {patience}",
        "--disable_tqdm True",
        f"--results_csv {results_csv}",
    ])
    
    return " ".join(cmd_parts)


def main(config_path="hyperparameters.json") -> None:
    config = load_config(config_path)
    global_config = config["global"]
    
    # グローバル設定を取得
    file_tag = global_config.get("file_tag", "DEFAULT")
    mode = global_config.get("mode", "HYPERPARAMETER_SEARCH")
    datasets = global_config["datasets"]
    models = global_config["models"]
    n_runs = global_config.get("n_runs", 20)
    epochs = global_config.get("epochs", 1000)
    early_stopping = global_config.get("early_stopping", True)
    patience = global_config.get("patience", 50)
    
    job_file_name = f"{file_tag}.txt"
    results_csv_name = f"{file_tag}.csv"
    info_file_name = f"{file_tag}.md"
    
    total_jobs = 0
    
    for dataset_name, model in itertools.product(datasets, models):
        # モデル固有のパラメータを取得
        params, model_param_names = get_params_for_model(config, model)
        
        # Create directory structure
        output_dir = f"jobs/{dataset_name}/{model}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate jobs for this dataset-model combination
        param_names = list(params.keys())
        param_lists = list(params.values())
        
        jobs_per_combo = 1
        for param_list in param_lists:
            jobs_per_combo *= len(param_list)
        
        total_jobs += jobs_per_combo
        
        # Write job file for each dataset-model combination
        output_file = os.path.join(output_dir, job_file_name)
        results_csv = os.path.join(output_dir, results_csv_name)
        with open(output_file, "w") as f:
            for param_values in itertools.product(*param_lists):
                # パラメータの辞書を作成
                params_dict = dict(zip(param_names, param_values))
                
                # n_runsの決定
                if mode == 'HYPERPARAMETER_SEARCH':
                    # HYPERPARAMETER_SEARCHのときは、Texas/Cornell/Wisconsinなら20、それ以外（Actor含む）は5
                    n_runs_value = 20 if dataset_name in ['Texas', 'Cornell', 'Wisconsin'] else 5
                else:
                    # HYPERPARAMETER_SEARCHでないときはN_RUNSを使用
                    n_runs_value = n_runs
                
                cmd = build_command(
                    dataset_name, model, params_dict, model_param_names,
                    n_runs_value, epochs, early_stopping, patience, results_csv
                )
                f.write(cmd + "\n")
        
        # Write info file
        info_file = os.path.join(output_dir, info_file_name)
        with open(info_file, "w") as f:
            f.write("# Experiment Configuration\n\n")
            f.write(f"**Dataset:** {dataset_name}\n\n")
            f.write(f"**Model:** {model}\n\n")
            f.write(f"**Total Jobs:** {jobs_per_combo}\n\n")
            f.write("## Hyperparameter Grid Search\n\n")
            f.write("| Parameter | Values |\n")
            f.write("|-----------|--------|\n")
            
            # Only write parameters that have multiple values (len > 1)
            param_display_names = {
                "label_noise_rates": "Label Noise Rate",
                "feature_missing_rates": "Feature Missing Rate",
                "feature_noise_rates": "Feature Noise Rate",
                "hidden_channels": "Hidden Channels",
                "num_layers": "Num Layers",
                "label_max_hops": "Label Max Hops",
                "label_temperature": "Label Temperature",
                "lp_alpha": "LP Alpha",
                "goodie_lamb": "Goodie Lambda",
                "fisf_num_iterations": "FISF Num Iterations",
                "fisf_alpha": "FISF Alpha",
                "fisf_beta": "FISF Beta",
                "fisf_gamma": "FISF Gamma",
                "fisf_mask_type": "FISF Mask Type",
            }
            
            for param_name, param_values in params.items():
                display_name = param_display_names.get(param_name, param_name)
                if len(param_values) > 1:
                    f.write(f"| {display_name} | {param_values} |\n")
            
            f.write("\n## Training Configuration\n\n")
            f.write(f"- **Epochs:** {epochs}\n")
            f.write(f"- **Early Stopping:** {early_stopping}\n")
            f.write(f"- **Patience:** {patience}\n")
            f.write(f"- **Number of Runs:** {n_runs}\n")
        
        print(f"Created {output_file} with {jobs_per_combo} jobs")
        print(f"Created {info_file} with configuration info")
    
    print(f"\nTotal jobs across all configurations: {total_jobs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate job files from hyperparameter configuration")
    parser.add_argument(
        "--config",
        type=str,
        default="hyperparameters.json",
        help="Path to hyperparameters JSON configuration file"
    )
    args = parser.parse_args()
    main(args.config)
