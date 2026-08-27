import sys

sys.path.append('../')
import torch
from utils.FramsticksGraphDataset import FramsticksGraphDataset
from utils.FramsticksPostProcessor import FramsticksPostProcessor
from pyprojroot import here
from src.models.KlejdaGAE.KlejdaGraphAutoencoder import KlejdaGraphAutoencoder
from src.models.KlejdaGAE.KlejdaVariationalGraphAutoencoder import KlejdaVariationalGraphAutoencoder
from src.models.NewGAE.GraphAutoencoder import GraphAutoencoder
from src.models.NewGAE.VariationalGraphAutoencoder import VariationalGraphAutoencoder
import yaml

# Konieczne? do modułu frams
import os
import sys

IS_NEW_APPROACH = False
IS_VGAE = False

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

# Przygotowanie checkpointów nauczonych autoenkoderów
project_dir = here()
if IS_NEW_APPROACH:
	checkpoints_dir = project_dir / 'notebooks' / 'checkpoints' / 'final_checkpoints' / 'new'
else:
	checkpoints_dir = project_dir / 'notebooks' / 'checkpoints' / 'final_checkpoints' / 'klejda'
checkpoint_gae = torch.load(checkpoints_dir / 'gae.ckpt')
checkpoint_vgae = torch.load(checkpoints_dir / 'vgae.ckpt')
# Przygotowanie konfiguracji dla gae

# TODO: Gae i Vgae mają taką samą konfigurację i tak ma docelowo być
configs_dir = project_dir / 'configs'
if IS_NEW_APPROACH:
	config_gae_path = configs_dir / 'gae_config.yaml'
	config_vgae_path = configs_dir / 'gae_config.yaml'
else:
	config_gae_path = configs_dir / 'klejda_gae_config.yaml'
	config_vgae_path = configs_dir / 'klejda_gae_config.yaml'

with open(config_gae_path) as f:
	config_gae = yaml.safe_load(f)
with open(config_vgae_path) as f:
	config_vgae = yaml.safe_load(f)

if IS_NEW_APPROACH:
	if IS_VGAE:
		autoencoder = VariationalGraphAutoencoder(config=config_vgae, frams_module=frams)
		autoencoder.load_state_dict(checkpoint_vgae['state_dict'])
	else:
		autoencoder = GraphAutoencoder(config=config_gae, frams_module=frams)
		autoencoder.load_state_dict(checkpoint_gae['state_dict'])
else:
	if IS_VGAE:
		autoencoder = KlejdaVariationalGraphAutoencoder(config=config_vgae, frams_module=frams)
		autoencoder.load_state_dict(checkpoint_vgae['state_dict'])
	else:
		autoencoder = KlejdaGraphAutoencoder(config=config_gae, frams_module=frams)
		autoencoder.load_state_dict(checkpoint_gae['state_dict'])
autoencoder.eval()

torch.set_printoptions(precision=3)
example_f0_individual = "//0\np:1.4554266080933014, 0.3524548501493414, -0.6360992981341942, fr=0.343, ing=0.251\np:0.2670381981844318, -0.2739245553662239, -1.0595551083384938, fr=0.842, ing=0.267\np:0.34743911607089006, -0.19253797145881857, 0.9369698617940858, fr=0.282, ing=0.451\np:1.4209797252347984, -0.7827931483818612, -0.6436342716281735, fr=0.527, ing=0.211\np:-0.20238053000679912, -0.3548171635176762, 2.8528414286303727, ing=0.485\nj:2, 4, rotstif=0.769\nj:3, 2, rotstif=0.965\nj:0, 2, stif=0.994, rotstif=0.967\nj:1, 2, rotstif=0.952\n"

print(f"Osobnik przed modyfikacją: {example_f0_individual}")

X_matrix, a_matrix, parts_num = FramsticksGraphDataset.parse_f0_to_matrices(example_f0_individual, 15)

print(f"macierz x: {X_matrix}")
print(f"macierz a: {a_matrix}")
print(f"Liczba elementów: {parts_num}")

X_matrix.unsqueeze_(0)
a_matrix.unsqueeze_(0)
with torch.no_grad():
	x_prime, a_prime = autoencoder.decode(autoencoder.encode(X_matrix, a_matrix))


x_prime.squeeze_(0)
a_prime.squeeze_(0)
print(f"macierz x_prime: {x_prime}")
print(f"macierz a_prime: {a_prime}")

parser = FramsticksPostProcessor()
_, parsed_to_f0, _, tries, _ = parser.process(x_prime, a_prime)

print(f"Osobnik przed: {example_f0_individual}")
print(f"Osobnik po: {parsed_to_f0}")
print(f"Liczba prób rekonstrukcji: {tries}")
