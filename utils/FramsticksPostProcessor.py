import numpy as np
import networkx as nx
import torch


class FramsticksPostProcessor:
	def __init__(self, max_joint_length: float = 2.0, max_iterations: int = 100, threshold: float = 0.05,
				 learning_rate: float = 0.1):
		self.max_len = max_joint_length
		self.max_it = max_iterations
		self.threshold = threshold
		self.lr = learning_rate  # Szybkość łagodnej naprawy

	def process(self, x_prime: torch.Tensor, a_prime: torch.Tensor):
		x = x_prime.detach().cpu().numpy()
		a = a_prime.detach().cpu().numpy()

		a_bin = (a > self.threshold).astype(int)
		a_sym = np.logical_or(a_bin, a_bin.T).astype(int)

		existing_nodes = []
		for i in range(a_sym.shape[0]):
			if a_sym[i, i] == 1 or np.sum(a_sym[i, :]) > (1 if a_sym[i, i] == 1 else 0):
				existing_nodes.append(i)

		if len(existing_nodes) == 0:
			return False, "", None, 0

		node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(existing_nodes)}

		G = nx.Graph()
		for new_idx, old_idx in enumerate(existing_nodes):
			features = x[old_idx].copy()
			G.add_node(
				new_idx,
				pos=features[:3],
				fr=features[3],
				ing=features[4]
			)

		for i in existing_nodes:
			for j in existing_nodes:
				if i < j and a_sym[i, j] == 1:
					G.add_edge(node_map[i], node_map[j])

		# 1. Weryfikacja: Zwróć błąd jeśli nie ma dokładnie JEDNEGO połączonego obiektu
		if not nx.is_connected(G):
			return False, "", G, 0

		self._fix_zero_length_joints(G)

		# 2. Łagodna naprawa
		is_successful, repair_tries = self._soft_repair_joints(G)

		# 3. Jeśli nie udało się naprawić (struktura była zbyt mocno zepsuta) - nie generujemy stringa
		if is_successful:
			f0_string = self._generate_f0_string(G)
		else:
			f0_string = ""

		return is_successful, f0_string, G, repair_tries

	def _fix_zero_length_joints(self, G: nx.Graph, epsilon: float = 0.01):
		for u, v in G.edges():
			pos_u = G.nodes[u]['pos']
			pos_v = G.nodes[v]['pos']
			if np.linalg.norm(pos_u - pos_v) < 1e-5:
				G.nodes[v]['pos'][0] += epsilon

	def _soft_repair_joints(self, G: nx.Graph) -> tuple[bool, int]:
		"""
		Działa jak system fizyczny (force-directed layout). Węzły połączone zbyt
		długą krawędzią zachowują się jak naciągnięta sprężyna, która je przyciąga.
		"""
		for iteration in range(self.max_it):
			max_current_len = 0.0
			displacements = {n: np.zeros(3) for n in G.nodes()}
			needs_repair = False

			for u, v in G.edges():
				pos_u = G.nodes[u]['pos']
				pos_v = G.nodes[v]['pos']

				diff = pos_u - pos_v
				dist = np.linalg.norm(diff)

				if dist > max_current_len:
					max_current_len = dist

				if dist > self.max_len:
					needs_repair = True
					excess = dist - self.max_len
					direction = diff / (dist + 1e-9)  # Kierunek od v do u

					# Węzły o mniejszej liczbie połączeń (np. końcówki ramion) stawiają
					# mniejszy opór i przesuwają się bardziej niż centralne węzły.
					deg_u = G.degree(u)
					deg_v = G.degree(v)
					total_deg = deg_u + deg_v

					weight_u = deg_v / total_deg
					weight_v = deg_u / total_deg

					# Łagodna siła przyciągania z uwzględnieniem "masy" i learning rate
					step_u = -direction * excess * self.lr * weight_u
					step_v = direction * excess * self.lr * weight_v

					displacements[u] += step_u
					displacements[v] += step_v

			if not needs_repair:
				return True, iteration  # Naprawa się powiodła przed wyczerpaniem limitu!

			# Symultaniczna aktualizacja wszystkich pozycji (zapobiega konfliktom)
			for n in G.nodes():
				G.nodes[n]['pos'] += displacements[n]

		# Weryfikacja końcowa - po wyczerpaniu limitu iteracji, czy odległości są znośne?
		# Dodajemy drobną tolerancję błędu floatów (1e-4)
		if len(G.edges()) > 0:
			final_max_len = max([np.linalg.norm(G.nodes[u]['pos'] - G.nodes[v]['pos']) for u, v in G.edges()])
		else:
			final_max_len = 0.0

		if final_max_len <= self.max_len + 1e-4:
			return True, self.max_it

		# Jeżeli pomimo 100 łagodnych ruchów kształt nadal jest niefizyczny - poddajemy się.
		return False, self.max_it

	def _generate_f0_string(self, G: nx.Graph) -> str:
		lines = ["//0"]
		for i in range(len(G.nodes)):
			pos = G.nodes[i]['pos']
			fr = G.nodes[i]['fr']
			ing = G.nodes[i]['ing']

			# Formatujemy floaty by uniknąć bardzo długich ciągów liczb
			lines.append(f"p:{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}, fr={fr:.3f}, ing={ing:.3f}")

		for u, v in G.edges():
			lines.append(f"j:{u}, {v}")

		return "\n".join(lines) + "\n"