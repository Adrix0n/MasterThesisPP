import numpy as np
import networkx as nx
import torch

class FramsticksPostProcessor:
	def __init__(self, max_joint_length: float = 2.0, max_iterations: int = 100, threshold: float = 0.05):
		self.max_len = max_joint_length
		self.max_it = max_iterations
		self.threshold = threshold

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
			return False, "", None, None

		node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(existing_nodes)}

		G = nx.Graph()
		for new_idx, old_idx in enumerate(existing_nodes):
			features = x[old_idx].copy()
			# x_prime ma teraz 5 cech. Rozdzielamy pozycję (indeksy 0,1,2)
			# od właściwości fizycznych (indeksy 3,4)
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

		self._fix_zero_length_joints(G)
		self._match_isolated_parts(G)
		is_valid = self._remove_too_long_joints(G)

		f0_string = self._generate_f0_string(G) if is_valid else ""

		return is_valid, f0_string, G

	def _fix_zero_length_joints(self, G: nx.Graph, epsilon: float = 0.01):
		for u, v in G.edges():
			pos_u = G.nodes[u]['pos']
			pos_v = G.nodes[v]['pos']
			if np.linalg.norm(pos_u - pos_v) < 1e-5:
				G.nodes[v]['pos'][0] += epsilon

	def _match_isolated_parts(self, G: nx.Graph):
		while not nx.is_connected(G):
			components = list(nx.connected_components(G))

			min_dist = float('inf')
			best_pair = None

			for i in range(len(components) - 1):
				for j in range(i + 1, len(components)):
					group1 = list(components[i])
					group2 = list(components[j])

					for u in group1:
						for v in group2:
							dist = np.linalg.norm(G.nodes[u]['pos'] - G.nodes[v]['pos'])
							if dist < min_dist:
								min_dist = dist
								best_pair = (u, v)
			if best_pair:
				G.add_edge(best_pair[0], best_pair[1])

	def _remove_too_long_joints(self, G: nx.Graph) -> bool:
		counter = 0

		while counter <= self.max_it:
			too_long_joints = []

			for u, v in G.edges():
				dist = np.linalg.norm(G.nodes[u]['pos'] - G.nodes[v]['pos'])
				if dist > self.max_len:
					too_long_joints.append((u, v, dist))

			if not too_long_joints:
				return True

			for u, v, dist in too_long_joints:
				degree_u = G.degree(u)
				degree_v = G.degree(v)

				p_less = u if degree_u <= degree_v else v
				p_more = v if p_less == u else u

				pos_less = G.nodes[p_less]['pos']
				pos_more = G.nodes[p_more]['pos']

				direction = pos_less - pos_more
				direction = direction / np.linalg.norm(direction)

				new_pos = pos_more + direction * (self.max_len - 1e-4)
				G.nodes[p_less]['pos'] = new_pos

			counter += 1
		return False

	def _generate_f0_string(self, G: nx.Graph) -> str:
		lines = ["//0"]

		for i in range(len(G.nodes)):
			pos = G.nodes[i]['pos']
			fr = G.nodes[i]['fr']
			ing = G.nodes[i]['ing']

			# Doklejamy fr= oraz ing= zgodnie z formatem Twojego zbioru danych
			lines.append(f"p:{pos[0]}, {pos[1]}, {pos[2]}, fr={fr}, ing={ing}")

		for u, v in G.edges():
			lines.append(f"j:{u}, {v}")

		return "\n".join(lines) + "\n"