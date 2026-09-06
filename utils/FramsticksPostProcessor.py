import numpy as np
import networkx as nx
import torch
from enum import Flag, auto

class PostProcessFlag(Flag):
	VALID = 0
	INVALID_ZERO_LENGTH_JOINTS = auto()
	INVALID_SUBGROUPS = auto()
	INVALID_TO_LONG_PARTS = auto()
	INVALID = auto()

class FramsticksPostProcessor:
	def __init__(self, max_joint_length: float = 2.0, max_iterations: int = 100, threshold: float = 0.5,
				 epsilon: float = 1e-5, repair=True):
		self.max_len = max_joint_length
		self.max_it = max_iterations
		self.threshold = threshold  # Próg określający, czy dane połączenie istnieje (1) czy nie istnieje (0_
		self.epsilon = epsilon
		self.repair = repair

	def process(self, x_prime: torch.Tensor, a_prime: torch.Tensor):
		x = x_prime.detach().cpu().numpy()
		a = a_prime.detach().cpu().numpy()

		# binaryzacja macierzy sąsiedztwa
		a_bin = (a > self.threshold).astype(int)
		# symetryzacja macierzy
		a_sym = np.logical_or(a_bin, a_bin.T).astype(int)
		# Wyzerowanie przekątnej
		# TODO: Rozważyć, czy powinno wyzerowywać przekątną. Działając w ten sposób, autoenkoder nie będzie zwracał uwagi na to, że tworzy niepoprawne rozwiązanie, bo z jego perspektywy macierz z wyzerowaną przekątną i nie wyzerowaną będzie taka sama
		# np.fill_diagonal(a_sym, 0)

		# wyznaczenie listy istniejących węzłów
		# Sprawdzany jest każdy wiersz, czy istnieje w nim jakaś jedynka, czyli połączenie z innym
		existing_nodes = []
		for i in range(a_sym.shape[0]):
			if np.any(a_sym[i, :]):
				existing_nodes.append(i)

		flags = PostProcessFlag.VALID

		# Brak węzłów, coś jest nie tak
		if len(existing_nodes) == 0:
			return False, "", None, 0, PostProcessFlag.INVALID

		G = nx.Graph()
		node_map = {}  # Słownik tłumaczący indeksy z macierzy na indeksy z listy
		new_idx = 0

		for old_idx in existing_nodes:
			node_map[old_idx] = new_idx

			features = x[old_idx].copy()

			G.add_node(
				new_idx,
				pos=features[:3]
			)
			new_idx += 1

		# Łączenie węzłów
		for i in existing_nodes:
			for j in existing_nodes:
				# Nie dublujemy połączeń poprzez warunek 1 < j
				if i < j and a_sym[i, j] == 1:
					# Wykorzystanie słownika mapującego indeksy
					G.add_edge(node_map[i], node_map[j])

		# Naprawa połączeń o zerowej długości
		if self._has_zero_length_joints(G) and self.repair:
			flags |= PostProcessFlag.INVALID_ZERO_LENGTH_JOINTS
			self._fix_zero_length_joints(G)

		# Naprawa rozdzielonych podgrup
		if not nx.is_connected(G) and self.repair:
			flags |= PostProcessFlag.INVALID_SUBGROUPS
			self._repair_isolated_parts(G)

		# Naprawa zbyt długich połączeń
		repair_tries = 0
		if self._has_too_long_joints(G) and self.repair:
			flags |= PostProcessFlag.INVALID_TO_LONG_PARTS
			is_successful, repair_tries = self._repair_too_long_joints(G)
			if not is_successful:
				flags |= PostProcessFlag.INVALID

		# Check final validity
		is_successful = (PostProcessFlag.INVALID not in flags)

		if is_successful:
			f0_string = self._generate_f0_string(G)
		else:
			f0_string = ""

		return is_successful, f0_string, G, repair_tries, flags

	def _has_zero_length_joints(self, G: nx.Graph) -> bool:
		for u, v in G.edges():
			# Norma wektorowa pozycji dwóch węzłów (Zwykła odległość Euklidesowa)
			if np.linalg.norm(G.nodes[u]['pos'] - G.nodes[v]['pos']) < self.epsilon:
				return True
		return False

	def _fix_zero_length_joints(self, G: nx.Graph):
		for u, v in G.edges():
			pos_u = G.nodes[u]['pos']
			pos_v = G.nodes[v]['pos']
			if np.linalg.norm(pos_u - pos_v) < self.epsilon:
				# Przesunięcie jednego z węzłów o sqrt(3) * epsilon
				G.nodes[v]['pos'][0] += self.epsilon
				G.nodes[v]['pos'][1] += self.epsilon
				G.nodes[v]['pos'][2] += self.epsilon

	def _repair_isolated_parts(self, G: nx.Graph):
		groups = list(nx.connected_components(G))

		while len(groups) > 1:
			possible_connections = []

			# Znajduje najkrótsze połączenie pomiędzy rozdzielonymi grupami
			for i in range(len(groups) - 1):
				for j in range(i + 1, len(groups)):
					min_dist = float('inf')
					best_edge = None

					# Iteracja po węzłach grup w poszukiwaniu najkrótszego odcinka
					for u in groups[i]:
						for v in groups[j]:
							dist = np.linalg.norm(G.nodes[u]['pos'] - G.nodes[v]['pos'])
							if dist < min_dist:
								min_dist = dist
								best_edge = (u, v)

					possible_connections.append((min_dist, i, j, best_edge))

			# Wybór najkrótszego połączenia
			possible_connections.sort(key=lambda item: item[0])
			_, i, j, connection = possible_connections[0]

			# Dodanie węzła
			G.add_edge(*connection)

			# Złączenie grup
			groups[i] = groups[i].union(groups[j])
			groups.pop(j)

	def _has_too_long_joints(self, G: nx.Graph) -> bool:
		return len(self._find_too_long_joints(G)) > 0

	def _find_too_long_joints(self, G: nx.Graph) -> list:
		too_long = []
		for u, v in G.edges():
			if np.linalg.norm(G.nodes[u]['pos'] - G.nodes[v]['pos']) > self.max_len:
				too_long.append((u, v))
		return too_long

	def _repair_too_long_joints(self, G: nx.Graph) -> tuple[bool, int]:
		counter = 0
		too_long_joints = self._find_too_long_joints(G)

		while len(too_long_joints) > 0 and counter < self.max_it:
			for p1, p2 in too_long_joints:
				# Wybierana jest część o mniejszej liczbie punktów
				if G.degree(p1) <= G.degree(p2):
					p_less, p_more = p1, p2
				else:
					p_less, p_more = p2, p1

				pos_less = G.nodes[p_less]['pos']
				pos_more = G.nodes[p_more]['pos']

				diff = pos_more - pos_less
				dist = np.linalg.norm(diff)

				# Przemieszczenie w stronę centrum
				if dist > self.max_len:
					excess = dist - self.max_len + 1e-5 # Dodanie małej części aby 'przestrzelić' i na pewno zmniejszyć odległość do mniejszej niż
					direction = diff / (dist + 1e-9)
					G.nodes[p_less]['pos'] = pos_less + (direction * excess)

			counter += 1
			too_long_joints = self._find_too_long_joints(G)

		# jeżeli po wszystkich iteracjach cały czas są problemy, to zwracamy jako invalid
		is_successful = len(too_long_joints) == 0
		return is_successful, counter

	def _generate_f0_string(self, G: nx.Graph) -> str:
		lines = ["//0"]
		for i in range(len(G.nodes)):
			pos = G.nodes[i]['pos']
			lines.append(f"p:{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}")

		for u, v in G.edges():
			lines.append(f"j:{u}, {v}")

		return "\n".join(lines) + "\n"