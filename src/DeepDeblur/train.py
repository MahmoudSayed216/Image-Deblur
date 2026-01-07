import argparse
import sys
import yaml
import os
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.transforms import functional as F
from .utils.transforms import apply_transforms
from .GoProDataset import GoProDataset
from .logger import Logger
from .CheckpointsHandler import CheckpointsHandler
from torch.nn import MSELoss
from torch.optim import AdamW
from .model.DeepDeblur import DeepDeblur
from .Metrics import PSNR, SSIM

def load_configs() -> tuple[dict, dict]:
    
    CONFIGS_YAML_FILE_PATH = os.path.join(os.path.dirname(__file__) ,"configs.yaml")
    SHARED_CONFIGS_YAML_FILE_PATH = os.path.join(os.path.dirname(__file__) ,"../shared_configs.yaml")
    
    
    with open(CONFIGS_YAML_FILE_PATH) as f:
        training_configs = yaml.safe_load(f)
    with open(SHARED_CONFIGS_YAML_FILE_PATH) as x:
        shared_configs = yaml.safe_load(x)

    return training_configs, shared_configs

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

def override_configs(training_configs: dict, shared_configs: dict, args: dict):

    if args["n_channels"] is not None:
        training_configs["model"]["n_channels"] = args["n_channels"]

    if args["lr"] is not None:
        training_configs["training"]["learning_rate"] = args["lr"]
    if args["patience"] is not None:
        training_configs["training"]["patience"] = args["patience"]
    if args["epochs"] is not None:
        training_configs["training"]["epochs"] = args["epochs"]
    if args["batch_size"] is not None:
        training_configs["training"]["batch_size"] = args["batch_size"]
    if args["start_epoch"] is not None:
        training_configs["training"]["start_epoch"] = args["start_epoch"]
    if args["save_every"] is not None:
        training_configs["training"]["save_every"] = args["save_every"]

    training_configs["checkpoint"]["continue"] = args["continue_from_checkpoint"]
    
    if args["checkpoint_id"] is not None:
        training_configs["checkpoint"]["id"] = args["checkpoint_id"]
    if args["checkpoint_type"] is not None:
        training_configs["checkpoint"]["type"] = args["checkpoint_type"]

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
    # print()
    return dataset_path, output_path

def create_training_environment(output_relative_path: str) -> str:
    BASE_DIR = Path(__file__).resolve().parent #! TRY os.path.dirname INSTEAD
    output_path_base = os.path.join(BASE_DIR, output_relative_path)
    print("OUTPUT PATH: ", output_path_base)
    if not os.path.exists(output_path_base):
        os.mkdir(os.path.join(output_path_base))
    
    session_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    session_path = os.path.join(output_path_base, session_id)
    os.mkdir(session_path)


    logs_folder = os.path.join(session_path, "logs")
    os.mkdir(path=logs_folder)
    weights_folder = os.path.join(session_path, "weights")
    os.mkdir(path=weights_folder)

    return session_path

def log_all_configs(logger, session_path, training_configs, shared_configs):
    key_val_pairs =  []
    def traverse_dict(d, parent_key=""):
        for key, value in d.items():
            full_key = f"{parent_key}-{key}" if parent_key else str(key)

            if isinstance(value, dict):
                traverse_dict(value, full_key)
            else:
                key_val_pairs.append(f"{full_key}: {value}")

    traverse_dict(training_configs)
    output = "\n".join(key_val_pairs)
    output = "\ntrain configs: "+"\n" + output
    logger.log(output)
    key_val_pairs.clear()

    traverse_dict(shared_configs)
    output = "\n".join(key_val_pairs)
    output = "\nshared configs: "+"\n" + output
    logger.log(output)
    key_val_pairs.clear()

    logger.log("session path: ", session_path, end="\n\n")

    # print(output)


#TODO: implement this function
def create_data_loaders(dataset_path: str, training_configs: dict, shared_configs: dict) -> tuple[DataLoader, DataLoader]:
    train_ds = GoProDataset(dataset_path, split="train", transforms=apply_transforms)
    test_ds = GoProDataset(dataset_path, split="test", transforms=None)
    device = shared_configs["device"]
    train_loader = DataLoader(dataset=train_ds, num_workers=5, shuffle=True, batch_size=training_configs["training"]["batch_size"], pin_memory=(device == "cuda"))
    test_loader = DataLoader(dataset=test_ds, num_workers=5, shuffle=True, batch_size=1, pin_memory=True)

    
    return train_loader, test_loader




#TODO: implement this function
def compute_test_metrics(model, loss_fn, device, test_loader) -> tuple[float, float]:
    model.eval()
    three_scales_mse = 0
    high_scale_mse = 0
    ssim_score = 0
    n_examples = 0
    with torch.no_grad():
        for i, ((_256b, _128b, _64b), (_256s, _128s, _64s)) in enumerate(test_loader):
            _256b = _256b.to(device)
            _128b = _128b.to(device)
            _64b = _64b.to(device)
            _256s = _256s.to(device)
            _128s = _128s.to(device)
            _64s = _64s.to(device)


            _256g, _128g, _64g = model([_256b, _128b, _64b])

            _256loss = loss_fn(_256g, _256s) / (256*256*3)
            _128loss = loss_fn(_128g, _128s) / (128*128*3)
            _64loss = loss_fn(_64g, _64s) / (64*64*3)
            total_loss = _256loss + _128loss + _64loss

            ssim_score += SSIM(_256g, _256s) 
            three_scales_mse+=total_loss.item()
            high_scale_mse+=_256loss.item()
            n_examples+=1

    three_scales_avg_mse = three_scales_mse/n_examples
    high_scale_avg_mse = high_scale_mse/n_examples
    psnr = PSNR(three_scales_mse, max_val=1)
    ssim = ssim_score/n_examples
    return three_scales_avg_mse, high_scale_avg_mse, psnr, ssim




    
#TODO: implement this function
def train(train_loader: DataLoader, test_loader: DataLoader, training_configs: dict, shared_configs: dict, session_path: str, logger: Logger):

    EPOCHS = training_configs["training"]["epochs"]
    LEARNING_RATE = training_configs["training"]["learning_rate"]
    SAVE_EVERY = training_configs["training"]["save_every"]
    START_EPOCH = training_configs["training"]["start_epoch"] #! gets overriden if ["checkpoint"]["continue"] is true
    DEVICE = shared_configs["device"]
    MODEL_NAME = training_configs["model"]["name"]
    weights_saving_path = os.path.join(session_path, "weights")
    cp_handler = CheckpointsHandler(save_every=SAVE_EVERY, increasing_metric=True, output_path=weights_saving_path)

    model = DeepDeblur(train_configs=training_configs, shared_configs=shared_configs).to(DEVICE)
    loss_fn = MSELoss()
    optim = AdamW(params=model.parameters(), lr=LEARNING_RATE)
    

    logger.log(f"Training {MODEL_NAME} starting for {EPOCHS-START_EPOCH+1} epochs, Learning rate = {LEARNING_RATE}, with AdamW optimizer") #! optim must be accessed through configs

    for epoch in range(START_EPOCH, EPOCHS+1):
        logger.log(f"Epoch: {epoch}")
        epoch_cummulative_loss = 0
        steps = 0
        for i, ((_256b, _128b, _64b), (_256s, _128s, _64s)) in enumerate(train_loader):
            
            _256b = _256b.to(DEVICE)
            _128b = _128b.to(DEVICE)
            _64b = _64b.to(DEVICE)
            _256s = _256s.to(DEVICE)
            _128s = _128s.to(DEVICE)
            _64s = _64s.to(DEVICE)

            _256g, _128g, _64g = model([_256b, _128b, _64b])
            
            _256loss = loss_fn(_256g, _256s) / (256*256*3)
            _128loss = loss_fn(_128g, _128s) / (128*128*3)
            _64loss = loss_fn(_64g, _64s) / (64*64*3)
            
            total_loss = _256loss + _128loss + _64loss
            total_loss.backward()
            optim.step()
            
            epoch_cummulative_loss+=total_loss.item()
            steps+=1
            if i % 20:
                logger.log(f"epoch: {epoch} big_step: {i}")
        
        #TODO: compute average epoch loss
        avg_train_mse = epoch_cummulative_loss/steps
        #TODO: compute test mse
        three_scales_test_mse, high_scale_test_mse, psnr, ssim = compute_test_metrics(model, loss_fn, DEVICE, test_loader)
        #TODO: log all numbers
        logger.log(f"Average Train MSE = {avg_train_mse:.3f}")
        logger.log(f"High scale test MSE = {high_scale_test_mse:.3f}")
        logger.log(f"3 scales test MSE = {three_scales_test_mse:.3f}")
        logger.log(f"PSNR: {psnr:.3f}")
        logger.log(f"SSIM: {ssim:.3f}")
        #TODO: Checkpoint Handler

        if cp_handler.check_save_every(epoch):
            logger.checkpoint(f"{SAVE_EVERY} epochs have passed, saving data in last.pth")
            cp_handler.save_model()
        if cp_handler.metric_has_improved(psnr):
            logger.checkpoint(f"metric has improved, saving data in best.pth")
            cp_handler.save_model()





def main():
    training_configs, shared_configs = load_configs()
    args = parse_args()
    
    if len(sys.argv) > 1:
        override_configs(training_configs, shared_configs, args)

    dataset_path, output_path = resolve_paths(shared_configs)

    session_path = create_training_environment(output_path)
    # logger = None
    logger = Logger(debug_mode=shared_configs["environment"]["debugger_active"], logs_folder_path=os.path.join(session_path, "logs"))

    #TODO: LOG ALL CONFIGS AND SESSION PATH BEFORE STARTING
    log_all_configs(logger=logger, session_path=session_path, training_configs=training_configs, shared_configs=shared_configs)
    # print(os.listdir(session_path))
    logger.debug("HI from train.py")
    logger.log("HI FROM train.py")
    logger.checkpoint("HI from train.py")
    logger.log("HI FROM train.py")
    logger.checkpoint("HI from train.py")
    logger.debug("HI from train.py")
    logger.checkpoint("HI from train.py")
    logger.log("HI FROM train.py")
    logger.checkpoint("HI from train.py")
    # * REVIEW THE TRANSFORMS DONE IN THE TRAINING SECTION
    train_loader, test_loader = create_data_loaders(dataset_path, training_configs, shared_configs)

    train(train_loader = train_loader,test_loader =  test_loader, training_configs=training_configs, shared_configs=shared_configs, session_path=session_path, logger=logger)


if __name__ == "__main__":

    main()