import argparse
import sys
import yaml
import os
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader


def load_configs() -> tuple[dict, dict]:
    
    CONFIGS_YAML_FILE_PATH = "configs.yaml"
    SHARED_CONFIGS_YAML_FILE_PATH = "../shared_configs.yaml"
    
    
    with open(CONFIGS_YAML_FILE_PATH) as f:
        train_configs = yaml.safe_load(f)
    with open(SHARED_CONFIGS_YAML_FILE_PATH) as x:
        shared_configs = yaml.safe_load(x)

    return train_configs, shared_configs

def parse_args() -> dict:

    parser = argparse.ArgumentParser("Model Training Configuration")

    #* Model
    parser.add_argument("--n_channels", type=int, default=None)

    #* Training
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--start-epoch", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)

    #* Checkpoint
    parser.add_argument("--continue-from-checkpoint", action="store_true")  # store_true is already False by default
    parser.add_argument("--checkpoint-id", type=str, default=None)
    parser.add_argument("--checkpoint-type", type=str, choices=["best", "last"], default=None)

    #* Debug / Environment
    parser.add_argument("--use-debugger", action="store_true")  # False if not given
    parser.add_argument("--kaggle", action="store_true")         # False if not given
    parser.add_argument("--dataset-base-kaggle", type=str, default=None)
    parser.add_argument("--dataset-base-local", type=str, default=None)
    parser.add_argument("--output-dir-kaggle", type=str, default=None)
    parser.add_argument("--output-dir-local", type=str, default=None)

    #* Device
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default=None)

    args = parser.parse_args()


    return args.__dict__

def override_configs(train_configs: dict, shared_configs: dict, args: dict):

    if args["n_channels"] is not None:
        train_configs["model"]["n_channels"] = args["n_channels"]

    if args["lr"] is not None:
        train_configs["training"]["learning_rate"] = args["lr"]
    if args["patience"] is not None:
        train_configs["training"]["patience"] = args["patience"]
    if args["epochs"] is not None:
        train_configs["training"]["epochs"] = args["epochs"]
    if args["batch_size"] is not None:
        train_configs["training"]["batch_size"] = args["batch_size"]
    if args["start_epoch"] is not None:
        train_configs["training"]["start_epoch"] = args["start_epoch"]
    if args["save_every"] is not None:
        train_configs["training"]["save_every"] = args["save_every"]

    train_configs["checkpoint"]["continue"] = args["continue_from_checkpoint"]
    
    if args["checkpoint_id"] is not None:
        train_configs["checkpoint"]["id"] = args["checkpoint_id"]
    if args["checkpoint_type"] is not None:
        train_configs["checkpoint"]["type"] = args["checkpoint_type"]

    shared_configs["use_debugger"] = args["use_debugger"] 
    shared_configs["kaggle"] = args["kaggle"]

    if args["dataset_base_kaggle"] is not None:
        shared_configs["environment"]["dataset_base"]["kaggle"] = args["dataset_base_kaggle"]
    if args["dataset_base_local"] is not None:
        shared_configs["environment"]["dataset_base"]["local"] = args["dataset_base_local"]
    
    if args["output_dir_kaggle"] is not None:
        shared_configs["environment"]["output_dir"]["kaggle"] = args["output_dir_kaggle"]
    if args["output_dir_local"] is not None:
        shared_configs["environment"]["output_dir"]["local"] = args["output_dir_local"]
    
    if args["device"] is not None:
        shared_configs["device"] = args["device"]

def resolve_paths(shared_configs: dict) -> tuple[str, str]:
    if shared_configs["environment"]["kaggle"]:
        dataset_path = shared_configs["environment"]["dataset_base"]["kaggle"]
        output_path = shared_configs["environment"]["output_dir"]["kaggle"]
    else:
        dataset_path = shared_configs["environment"]["dataset_base"]["local"]
        output_path = shared_configs["environment"]["output_dir"]["local"]

    return dataset_path, output_path

def create_training_environment(output_relative_path: str) -> str:
    BASE_DIR = Path(__file__).resolve().parent
    output_path_base = os.path.join(BASE_DIR, output_relative_path)
    if not os.path.exists(output_path_base):
        os.mkdir(os.path.join(output_path_base))
    
    session_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    session_path = os.path.join(output_path_base, session_id)
    os.mkdir(session_path)
    print(session_path)

    return session_path


#TODO: implement this function
def create_data_loaders(dataset_path: str) -> tuple[DataLoader, DataLoader]:
    pass
#TODO: implement this function
def compute_validation_metrics(model, test_loader) -> tuple[float, float]:
    pass
#TODO: implement this function
def train(train_configs, shared_configs, session_path):
    pass


def main():
    train_configs, shared_configs = load_configs()
    args = parse_args()
    
    if len(sys.argv) > 1:
        override_configs(train_configs, shared_configs, args)

    dataset_path, output_path = resolve_paths(shared_configs)

    session_path = create_training_environment(output_path)

    #! REVIEW THE TRANSFORMS DONE IN THE TRAINING SECTION
    train_loader, test_loader = create_data_loaders(dataset_path)

    train(train_loader, test_loader, session_path)



if __name__ == "__main__":
    main()