#!/usr/bin/env python
# coding: utf-8

# # Opis
# 
# Krótki notebook, który pozwala przetestować działanie różnych elementów implementacyjnych w szybki sposób.

# # Importy

# In[3]:


IS_NEW_APPROACH = True
IS_VGAE = True
IS_SLURM = False


# In[4]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
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
	config_gae_path = configs_dir / 'gae_config_large.yaml'
else:
	config_gae_path = configs_dir / 'klejda_gae_config.yaml'

with open(config_gae_path) as f:
    config = yaml.safe_load(f)

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
    num_workers=4,
    persistent_workers=True,
	drop_last=True
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=config['batch_size'],
    shuffle=False,
    num_workers=4,
    persistent_workers=True,
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
	if IS_NEW_APPROACH:
		run_name = "GAE_new_test"
	else:
		run_name = "GAE_klejda_test"
	wandb_logger = WandbLogger(project="Framsticks-GAE", name=run_name, save_dir = config["save_dir"])

	trainer = pl.Trainer(
		precision="bf16-mixed",
		callbacks=[accumulator],
	    max_epochs=config['max_epochs'],
	    logger=wandb_logger,
	    log_every_n_steps=5,
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
	if IS_NEW_APPROACH:
		run_name = "VGAE_new_test"
	else:
		run_name = "VGAE_klejda_test"
	wandb_logger = WandbLogger(project="Framsticks-VGAE", name=run_name, save_dir = config["save_dir"])
	trainer = pl.Trainer(
	    max_epochs=config['max_epochs'],
	    logger=wandb_logger,
		callbacks=[accumulator],
	    enable_progress_bar=True,
		log_every_n_steps=50,
	    accelerator="auto",
	    devices=1
	)

	trainer.fit(modelVGAE, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
	wandb.finish()

