import torch
import wandb

wandb.init(project="bird_species_classification", entity="karanjakhar")
CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'epochs': 100000,
    'learning_rate': 0.001,
    'momentum': 0.9,
    'weight_decay':0.0001,
    'save_weights_epoch': 100,
    'batch_size':8,
    'best_validation_accuracy': 0,
    'pretrained_model_path': '../model/last.pth',
    'use_pretrained': False
}
wandb.config = CONFIG