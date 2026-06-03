import torch
from torch.utils.data import Dataset


class FramsticksGraphDataset(Dataset):
	def __init__(self, genotypes: list[str], max_nodes: int = 15):
		self.genotypes = genotypes
		self.max_nodes = max_nodes

	def __len__(self):
		return len(self.genotypes)

	def __getitem__(self, idx):
		genotype_str = self.genotypes[idx]
		return self.parse_f0_to_matrices(genotype_str, self.max_nodes)

	@staticmethod
	def parse_f0_to_matrices(f0_str: str, max_nodes: int):
		x_matrix = torch.full((max_nodes, 3), -1.0, dtype=torch.float32)
		a_matrix = torch.zeros((max_nodes, max_nodes), dtype=torch.float32)

		lines = f0_str.strip().split('\n')

		part_idx = 0

		for line in lines:
			line = line.strip()
			if not line or line.startswith('#') or line.startswith('//') or line.startswith('n:') or line.startswith(
					'c:'):
				continue

			if line.startswith('p:'):
				if part_idx >= max_nodes:
					raise ValueError(f"Znaleziono więcej niż {max_nodes} części w genotypie!")

				props_str = line[2:].strip()
				coords = [0.0, 0.0, 0.0]

				if props_str:
					parts = props_str.split(',')
					for i, prop in enumerate(parts):
						prop = prop.strip()
						if not prop: continue

						if '=' in prop:
							key, val = prop.split('=', 1)
							key = key.strip()
							try:
								if key == 'x':
									coords[0] = float(val)
								elif key == 'y':
									coords[1] = float(val)
								elif key == 'z':
									coords[2] = float(val)
							except ValueError:
								pass
						else:
							try:
								if i < 3: coords[i] = float(prop)
							except ValueError:
								pass
				x_matrix[part_idx] = torch.tensor(coords, dtype=torch.float32)
				a_matrix[part_idx, part_idx] = 1.0
				part_idx += 1

			elif line.startswith('j:'):
				props_str = line[2:].strip()
				if not props_str:
					continue

				p1, p2 = -1, -1
				parts = props_str.split(',')

				for i, prop in enumerate(parts):
					prop = prop.strip()
					if not prop: continue

					if '=' in prop:
						key, val = prop.split('=', 1)
						key = key.strip()
						try:
							if key == 'p1':
								p1 = int(val)
							elif key == 'p2':
								p2 = int(val)
						except ValueError:
							pass
					else:
						try:
							if i == 0:
								p1 = int(prop)
							elif i == 1:
								p2 = int(prop)
						except ValueError:
							pass
				if 0 <= p1 < max_nodes and 0 <= p2 < max_nodes:
					a_matrix[p1, p2] = 1.0
					a_matrix[p2, p1] = 1.0
		return x_matrix, a_matrix