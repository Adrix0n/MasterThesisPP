import torch
from torch.utils.data import Dataset


class FramsticksGraphDataset(Dataset):
	def __init__(self, genotypes: list[dict], max_nodes: int = 15):
		self.genotypes = genotypes
		self.max_nodes = max_nodes

	def __len__(self):
		return len(self.genotypes)

	def __getitem__(self, idx):
		genotype_str = self.genotypes[idx]['genotype']
		x, adj, parts_num = self.parse_f0_to_matrices(genotype_str, self.max_nodes)
		properties_dict = {
			"parts_num": torch.tensor(parts_num, dtype=torch.float64),
			"fitness": torch.tensor(self.genotypes[idx]['fitness'], dtype=torch.float64),
			"genotype": genotype_str
		}
		return x, adj, properties_dict

	@staticmethod
	def parse_f0_to_matrices(f0_str: str, max_nodes: int):
		# Wierzchołki teraz mają 3 cechy: [x, y, z]
		x_matrix = torch.zeros((max_nodes, 3), dtype=torch.float32)
		
		# Macierz sąsiedztwa (1.0 = jest krawędź, 0.0 = brak krawędzi)
		a_matrix = torch.zeros((max_nodes, max_nodes), dtype=torch.float32)

		lines = f0_str.strip().split('\n')
		part_idx = 0

		for line in lines:
			line = line.strip()
			if not line or line[0] in ('#', '/') or line.startswith(('n:', 'c:')):
				continue

			if line.startswith('p:'):
				if part_idx >= max_nodes:
					raise ValueError(f"Znaleziono więcej niż {max_nodes} części w genotypie!")

				props_str = line[2:].strip()
				props = {}
				unnamed_args = []

				# Rozdzielenie parametrów zdefiniowanych wprost od tych ze znakiem '='
				if props_str:
					for prop in props_str.split(','):
						if '=' in prop:
							key, val = prop.split('=', 1)
							props[key.strip()] = val.strip()
						else:
							unnamed_args.append(prop.strip())

				# Pobieranie współrzędnych
				x = float(unnamed_args[0]) if len(unnamed_args) > 0 else float(props.get('x', 0.0))
				y = float(unnamed_args[1]) if len(unnamed_args) > 1 else float(props.get('y', 0.0))
				z = float(unnamed_args[2]) if len(unnamed_args) > 2 else float(props.get('z', 0.0))

				# Pobieranie nowych cech z domyślną wartością 0.0 w przypadku ich braku
				# fr = float(props.get('fr', 0.0))
				# ing = float(props.get('ing', 0.0))

				x_matrix[part_idx] = torch.tensor([x, y, z], dtype=torch.float32)

				# a_matrix[part_idx, part_idx] = 1.0

				part_idx += 1

			elif line.startswith('j:'):
				props_str = line[2:].strip()
				if not props_str:
					continue

				props = {}
				unnamed_args = []

				for prop in props_str.split(','):
					if '=' in prop:
						key, val = prop.split('=', 1)
						props[key.strip()] = val.strip()
					else:
						unnamed_args.append(prop.strip())

				try:
					p1 = int(unnamed_args[0]) if len(unnamed_args) > 0 else int(props.get('p1', -1))
					p2 = int(unnamed_args[1]) if len(unnamed_args) > 1 else int(props.get('p2', -1))

					if 0 <= p1 < max_nodes and 0 <= p2 < max_nodes:
						a_matrix[p1, p2] = 1.0
						a_matrix[p2, p1] = 1.0
				except (ValueError, IndexError):
					pass

		x_matrix, a_matrix = FramsticksGraphDataset.sort_matrices_ascending(x_matrix, a_matrix)
		# Zwracamy 3 macierze oraz liczbę węzłów
		return x_matrix, a_matrix, part_idx

	@staticmethod
	def sort_matrices_ascending(x_unsorted, a_unsorted):
		# Opakowujemy wiersze X w indeksy
		x_unsorted_with_indices = list(enumerate(x_unsorted.tolist()))

		# Sortujemy rosnąco po kolei względem X, Y, Z
		x_sorted_with_indices = sorted(
			x_unsorted_with_indices,
			key=lambda item: (item[1][0], item[1][1], item[1][2])
		)

		sorted_indices = [item[0] for item in x_sorted_with_indices]
		idx_tensor = torch.tensor(sorted_indices, dtype=torch.long, device=x_unsorted.device)

		x_sorted = x_unsorted[idx_tensor]

		# Macierz A należy posortować zarówno po wierszach jak i po kolumnach
		a_sorted = a_unsorted[idx_tensor][:, idx_tensor]

		return x_sorted, a_sorted