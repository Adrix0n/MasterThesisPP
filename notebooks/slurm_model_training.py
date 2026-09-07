#!/usr/bin/env python
# coding: utf-8

# # Opis
# 
# NOtebook służący przygotowaniu testów do wykoania w slurmie

# # Importy

# In[3]:


IS_NEW_APPROACH = True
IS_SLURM = False


# In[4]:


import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import GradientAccumulationScheduler
import yaml
import sys
import tqdm
import wandb
import json
from pyprojroot import here
import argparse
sys.path.append('../') # Dodajemy katalog wyżej, żeby src był widoczny
from src.models.KlejdaGAE.KlejdaGraphAutoencoder import KlejdaGraphAutoencoder
from src.models.KlejdaGAE.KlejdaVariationalGraphAutoencoder import KlejdaVariationalGraphAutoencoder
from src.models.NewGAE.GraphAutoencoder import GraphAutoencoder
from src.models.NewGAE.VariationalGraphAutoencoder import VariationalGraphAutoencoder
from utils.FramsticksGraphDataset import FramsticksGraphDataset
from src.other.QuadraticGradientAccumulationScheduler import QuadraticGradientAccumulationScheduler


# In[6]:


import os
import sys
current_dir = os.getcwd()
framspy_path = os.path.abspath(os.path.join(current_dir, '..', 'external', 'framspy'))
if framspy_path not in sys.path:
	sys.path.insert(0, framspy_path)
with open("../configs/final_evolution_config.yaml", 'r') as f:
	evolution_config = yaml.safe_load(f)
import frams

frams.init(
    evolution_config['frams_path']
)


# In[ ]:


parser = argparse.ArgumentParser(description="Trening modelu na różnych konfiguracjach")

parser.add_argument(
    "--model_type",
    type=str,
    choices=["gae", "vgae"],
    default="gae",
    help="Zwykły 'gae', lub wariacyjny 'vgae'"
)

parser.add_argument(
    "--locality_loss_type",
    type=str,
    choices=["parts_num", "fitness"], #TODO: Dodać jeszcze silimality, jak już będzie szybciej działać
    default="fitness",
    help="Typ locality loss"
)

parser.add_argument(
    "--latent_dim",
    type=int,
    choices=[3, 10, 15, 100],
    default=3,
    help="Rozmiar przestrzeni ukrytej"
)

parser.add_argument(
    "--layers_config",
    type=str,
    choices=["small", "large"],
    default="small",
    help="Wariant głębokości sieci: 'small' lub 'large'"
)
parser.add_argument(
	"--exp_id",
	type=int,
	default=0,
	help="ID eksperymentu z siatki"
)
parser.add_argument("--run_name", type=str, default="slurm", help="Własna nazwa eksperymentu w WandB")
parser.add_argument("--group", type=str, default="slurm", help="Grupa eksperymentów w WandB")

args = parser.parse_args()


# # Przetwarzanie

# ## Załadowanie danych i przygotowanie do przetwarzania

# In[7]:


# dataset = FramsticksDummyDataset(num_samples=1000)
torch.set_float32_matmul_precision('medium')
genotypes = []
with open("../results/sampled_best_individuals_new_mini_merged.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line.strip())
        genotypes.append(obj)

if IS_SLURM:
    project_dir = Path("/home/inf151848/MasterThesisPP")
else:
    project_dir = here()

configs_dir = project_dir / 'configs'
if IS_NEW_APPROACH:
	config_gae_path = configs_dir / 'gae_config_final.yaml'
else:
	config_gae_path = configs_dir / 'klejda_gae_config.yaml'

with open(config_gae_path) as f:
    config = yaml.safe_load(f)

config["model_type"] = args.model_type
config["latent_dim"] = args.latent_dim
config["layers_config"] = args.layers_config
config["locality_loss_type"] = args.locality_loss_type
config["exp_id"] = args.exp_id
IS_VGAE = config["model_type"] == "vgae"

# Wczytanie dodatkowego configu od warstw modelu
config_layers = configs_dir
if config['layers_config'] == "small":
	config_layers = config_layers / "layers_small.yaml"
else:
	config_layers = config_layers / "layers_large.yaml"
with open(config_layers, "r") as f:
	config_layers_dict = yaml.safe_load(f)

for k in config_layers_dict.keys():
	config[k] = config_layers_dict[k]

dataset = FramsticksGraphDataset(genotypes,config["max_nodes"])

dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size
print("train_size:", train_size)
print("val_size:", val_size)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Akumulator, mający za zadanie zmieniać rozmiar batcha wraz z postępującym uczeniem modelu
# W początkowej fazie rozmiar jest mniejszy, wtedy bowiem model lepiej naucza się jak torzyć macierze X i A
# w późniejszym etapie rozmiar batcha wzrasta, aby skupić się poprawnym zmniejszeniu locality loss

if IS_NEW_APPROACH and config['use_accumulator']:
	accumulator = QuadraticGradientAccumulationScheduler(1,16,5,config['max_epochs'])
else:
	accumulator = GradientAccumulationScheduler(scheduling={
	    0:1,
	})

train_dataloader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=0,
    persistent_workers=False,
	drop_last=True
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=0,
    persistent_workers=False,
	drop_last=True
)

wandb.login()


# ## GAE

# In[44]:


if not IS_VGAE:
	if IS_NEW_APPROACH:
		modelGAE = GraphAutoencoder(config, frams_module=frams)
	else:
		modelGAE = KlejdaGraphAutoencoder(config, frams_module=frams)
	wandb.finish()


# In[45]:


if not IS_VGAE:
	wandb_logger = WandbLogger(project="Framsticks-MasterThesis", name=args.run_name, save_dir = config["save_dir"],tags=[args.model_type, args.layers_config, f"lat_{args.latent_dim}", f"loc_{args.locality_loss_type}"])
	wandb_logger.log_hyperparams(config)
	trainer = pl.Trainer(
	    max_epochs=config['max_epochs'],
	    logger=wandb_logger,
		callbacks=[accumulator],
	    enable_progress_bar=False,
		log_every_n_steps=10,
	    accelerator="auto",
	    devices=1
	)
	trainer.fit(modelGAE, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
	wandb.finish()


# ## VGAE

# In[8]:


if IS_VGAE:
	if IS_NEW_APPROACH:
		modelVGAE = VariationalGraphAutoencoder(config, frams_module=frams)
	else:
		modelVGAE = KlejdaVariationalGraphAutoencoder(config, frams_module=frams)
	wandb.finish()


# In[9]:


if IS_VGAE:
	wandb_logger = WandbLogger(project="Framsticks-MasterThesis", name=args.run_name, save_dir = config["save_dir"], tags=[args.model_type, args.layers_config, f"lat_{args.latent_dim}", f"loc_{args.locality_loss_type}"])
	wandb_logger.log_hyperparams(config)
	trainer = pl.Trainer(
	    max_epochs=config['max_epochs'],
	    logger=wandb_logger,
		callbacks=[accumulator],
	    enable_progress_bar=False,
		log_every_n_steps=10,
	    accelerator="auto",
	    devices=1
	)

	trainer.fit(modelVGAE, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
	wandb.finish()

