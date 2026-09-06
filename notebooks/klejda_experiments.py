#!/usr/bin/env python
# coding: utf-8

# # Description
# 
# W tym notatniku przeprowadzane są wszelkie eksperymenty, zarówno dla autoenkodera wariacyjnego i nie wariacyjnego, dla wszystkich członów funkcji straty, w wersji z douczaniem i bez (łącznie 12 eksperymentów)

# # Imports

# In[51]:


IS_NEW_APPROACH = True


# In[52]:


import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import RichProgressBar
import yaml
import sys
import os
import tqdm
import wandb
import json
import random

sys.path.append('../')  # Dodajemy katalog wyżej, żeby src był widoczny
from src.models.KlejdaGAE.KlejdaGraphAutoencoder import KlejdaGraphAutoencoder
from src.models.KlejdaGAE.KlejdaVariationalGraphAutoencoder import KlejdaVariationalGraphAutoencoder
from src.models.NewGAE.GraphAutoencoder import GraphAutoencoder
from src.models.NewGAE.VariationalGraphAutoencoder import VariationalGraphAutoencoder
from utils.FramsticksGraphDataset import FramsticksGraphDataset
from pyprojroot import here

current_dir = os.getcwd()
framspy_path = os.path.abspath(os.path.join(current_dir, '..', 'external', 'framspy'))
if framspy_path not in sys.path:
	sys.path.insert(0, framspy_path)
from FramsticksLib import FramsticksLib
from deap import tools, algorithms
import yaml
from src.deap.deap_setup import prepare_native_toolbox, prepare_cmaes_toolbox
from src.deap.constraints import is_feasible_fitness_criteria
from src.deap.save_and_load_results import save_genotypes_json
from src.deap.custom_ea_algorithms import run_cma_es_with_validation
from utils.FramsticksGraphDataset import FramsticksGraphDataset
from src.deap.AutoencoderEvaluator import AutoencoderEvaluator
import numpy as np
import frams
from copy import deepcopy

import time


# # Pre-processing

# In[53]:


project_dir = here()
# Przygotowanie checkpointów nauczonych autoenkoderów
checkpoints_dir = project_dir / 'notebooks' / 'checkpoints' / 'final_checkpoints'
if IS_NEW_APPROACH:
	checkpoints_dir = checkpoints_dir / 'new'
else:
	checkpoints_dir = checkpoints_dir / 'klejda'

checkpoint_gae = torch.load(checkpoints_dir / 'gae.ckpt')
checkpoint_vgae = torch.load(checkpoints_dir / 'vgae.ckpt')

CONTINUAL_TRAINING_DATASET_VERSION = 'remove_worse'

wandb.login()


# In[54]:


# Przygotowanie konfiguracji dla gae
configs_dir = project_dir / 'configs'
if IS_NEW_APPROACH:
	config_gae_path = configs_dir / 'gae_config_large.yaml'
else:
	config_gae_path = configs_dir / 'klejda_gae_config.yaml'
with open(config_gae_path) as f:
	config_gae = yaml.safe_load(f)
	config_vgae = config_gae


# In[55]:


# Wersja CMA-ES
with open("../configs/final_evolution_config.yaml", 'r') as f:
	evolution_config = yaml.safe_load(f)
frams.init(
	evolution_config['frams_path']
)
frams_lib = FramsticksLib(evolution_config['frams_path'], evolution_config['frams_lib'], evolution_config['sim_file'])
toolbox = prepare_cmaes_toolbox(frams_lib, evolution_config)

hof = tools.HallOfFame(evolution_config['hof_size'])
stats = tools.Statistics(lambda ind: ind.fitness.values)


def safe_filter_feasible(func, criteria):
	feasible_fits = list(filter(is_feasible_fitness_criteria, criteria))
	if len(feasible_fits) == 0:
		return np.nan
	return round(func(feasible_fits), 3)


stats.register("min", lambda fit: safe_filter_feasible(np.min, fit))
stats.register("avg", lambda fit: safe_filter_feasible(np.mean, fit))
stats.register("max", lambda fit: safe_filter_feasible(np.max, fit))


# # Experiments

# In[56]:


def update_genotypes_buffer(current_genotypes, new_genotypes, mode, max_size):
	if mode == 'append':
		return current_genotypes + new_genotypes

	combined = current_genotypes + new_genotypes

	if len(combined) <= max_size:
		return combined

	num_to_remove = len(combined) - max_size

	if mode == 'remove_worse':
		combined.sort(key=lambda x: x['fitness'], reverse=True)
		return combined[:max_size]

	elif mode == 'remove_last':
		return combined[num_to_remove:]

	elif mode == 'remove_random':
		return random.sample(combined, max_size)
	else:
		raise ValueError(
			f"Nieznana strategia: {mode}. Dostępne: 'append', 'remove_worse', 'remove_last', 'remove_random'")


# ## GAE

# In[7]:


if IS_NEW_APPROACH:
	gae_non_cyclic = GraphAutoencoder(config=config_gae, frams_module=frams).double()
else:
	gae_non_cyclic = KlejdaGraphAutoencoder(config=config_gae, frams_module=frams).double()
gae_non_cyclic.load_state_dict(checkpoint_gae['state_dict'])
gae_non_cyclic.eval()
evaluator = AutoencoderEvaluator(gae_non_cyclic, frams_lib, evolution_config['opt_criteria'], evolution_config)
toolbox.register("evaluate", evaluator)


# ### Trained once

# In[8]:


pop, log, reconstruction_ratio, _ = run_cma_es_with_validation(
	toolbox,
	ngen=evolution_config['generations'],
	stats=stats,
	halloffame=hof,
	verbose=True
)

print(f"\nNajlepszy fitness w HoF: {hof[0].fitness.values[0]}")
save_genotypes_json(evolution_config["result_filepath"], hof)


# ### Continual training

# In[39]:


with open("../configs/klejda_gae_config.yaml") as f:
    config_gae = yaml.safe_load(f)

loaded_genotypes = []
with open("../results/sampled_best_individuals_new_mini_merged.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line.strip())
        loaded_genotypes.append(obj)

genotypes = deepcopy(loaded_genotypes)
BUFFER_MAX_SIZE = len(genotypes)

for i in range(3):
    toolbox = prepare_cmaes_toolbox(frams_lib, evolution_config)
    toolbox.register("evaluate", evaluator)

    pop, log, created_individuals = run_cma_es_with_validation(
        toolbox,
        ngen=evolution_config['generations'],
        stats=stats,
        halloffame=hof,
        verbose=True,
    )

    new_inds = [{'genotype': ind.genotype, 'fitness': list(ind.fitness.values)} for ind in created_individuals]

    print(f"Dodatkowi osobnicy: {len(new_inds)}")
    fitnesses = [ind['fitness'][0] if isinstance(ind['fitness'], (list, tuple)) else ind['fitness'] for ind in genotypes]
    print(f"Statystyki przed: min: {min(fitnesses)}, max: {max(fitnesses)}, mean: {sum(fitnesses)/len(fitnesses)}")

    genotypes = update_genotypes_buffer(
        current_genotypes=genotypes,
        new_genotypes=new_inds,
        mode=CONTINUAL_TRAINING_DATASET_VERSION,
        max_size=BUFFER_MAX_SIZE
    )
    fitnesses = [ind['fitness'][0] if isinstance(ind['fitness'], (list, tuple)) else ind['fitness'] for ind in genotypes]
    print(f"Statystyki po: min: {min(fitnesses)}, max: {max(fitnesses)}, mean: {sum(fitnesses)/len(fitnesses)}")

    dataset = FramsticksGraphDataset(genotypes, config_gae["max_nodes"])

    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    wandb_logger = WandbLogger(
        project="Framsticks-GAE",
        name=f"GAE-Continual-{CONTINUAL_TRAINING_DATASET_VERSION}-Iter-{i}",
        save_dir=config_gae["save_dir"]
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        drop_last=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        drop_last=True
    )

    trainer = pl.Trainer(
        max_epochs=config_gae["supp_training_epochs"],
        logger=wandb_logger,
        log_every_n_steps=5,
        accelerator="auto",
        devices=1
    )

    gae_non_cyclic.train()
    trainer.fit(gae_non_cyclic, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    gae_non_cyclic.eval()
    wandb.finish()


# ## VGAE

# ### Trained once

# In[57]:


if IS_NEW_APPROACH:
	vgae_non_cyclic = VariationalGraphAutoencoder(config=config_gae, frams_module=frams)
else:
	vgae_non_cyclic = KlejdaVariationalGraphAutoencoder(config=config_gae, frams_module=frams)
vgae_non_cyclic.load_state_dict(checkpoint_vgae['state_dict'])
vgae_non_cyclic.eval()
evaluator = AutoencoderEvaluator(vgae_non_cyclic, frams_lib, evolution_config['opt_criteria'], evolution_config)
toolbox.register("evaluate", evaluator)


# In[10]:


pop, log, reconstruction_ratio, _ = run_cma_es_with_validation(
	toolbox,
	ngen=evolution_config['generations'],
	stats=stats,
	halloffame=hof,
	verbose=True
)

print(f"\nNajlepszy fitness w HoF: {hof[0].fitness.values[0]}")
save_genotypes_json(evolution_config["result_filepath"], hof)


# ### Continual training

# Wersja z douczaniem rozdziela się na różne możliwości:
# - Stały rozmiar zbioru douczającego, w którym osobniki:
# 	- Podmieniane są na zasadzie: słabsze odpadają
# 	- Podmieniane są na zasadzie: starsze odpadają
# 	- Podmieniane są na zasadzie: losowe odpadają
# - Zwiększający się rozmiar zbioru douczającego

# In[58]:


loaded_genotypes = []
with open("../results/sampled_best_individuals_new_mini_merged.jsonl", "r", encoding="utf-8") as f:
	for line in f:
		obj = json.loads(line.strip())
		loaded_genotypes.append(obj)

genotypes = deepcopy(loaded_genotypes)
BUFFER_MAX_SIZE = len(genotypes)

for i in range(3):
	toolbox = prepare_cmaes_toolbox(frams_lib, evolution_config)
	toolbox.register("evaluate", evaluator)

	pop, log, created_individuals = run_cma_es_with_validation(
		toolbox,
		ngen=evolution_config['generations'],
		stats=stats,
		halloffame=hof,
		verbose=True,
	)

	new_inds = [{'genotype': ind.genotype, 'fitness': list(ind.fitness.values)} for ind in created_individuals]

	print(f"Dodatkowi osobnicy: {len(new_inds)}")
	fitnesses = [ind['fitness'] for ind in genotypes]
	print(f"Statystyki przed: min: {min(fitnesses)}, max: {max(fitnesses)}, mean: {sum(fitnesses)/len(fitnesses)}")

	genotypes = update_genotypes_buffer(
		current_genotypes=genotypes,
		new_genotypes=new_inds,
		mode=CONTINUAL_TRAINING_DATASET_VERSION,
		max_size=BUFFER_MAX_SIZE
	)
	fitnesses = [ind['fitness'] for ind in genotypes]
	print(f"Statystyki po: min: {min(fitnesses)}, max: {max(fitnesses)}, mean: {sum(fitnesses)/len(fitnesses)}")



	dataset = FramsticksGraphDataset(genotypes, config_vgae["max_nodes"])

	dataset_size = len(dataset)
	train_size = int(0.8 * dataset_size)
	val_size = dataset_size - train_size
	train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

	wandb_logger = WandbLogger(
		project="Framsticks-VGAE",
		name=f"VGAE-Continual-{CONTINUAL_TRAINING_DATASET_VERSION}-Iter-{i}",
		save_dir=config_vgae["save_dir"]
	)

	train_dataloader = DataLoader(
		train_dataset,
		batch_size=256,
		shuffle=True,
		num_workers=4,
		persistent_workers=True,
		drop_last=True
	)

	val_dataloader = DataLoader(
		val_dataset,
		batch_size=256,
		shuffle=False,
		num_workers=4,
		persistent_workers=True,
		drop_last=True
	)

	trainer = pl.Trainer(
		max_epochs=config_vgae["supp_training_epochs"],
		logger=wandb_logger,
		log_every_n_steps=5,
		accelerator="auto",
		devices=1
	)

	vgae_non_cyclic.train()
	trainer.fit(vgae_non_cyclic, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
	vgae_non_cyclic.eval()
	wandb.finish()


# # Results
