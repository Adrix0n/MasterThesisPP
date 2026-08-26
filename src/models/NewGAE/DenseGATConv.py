import torch
import torch.nn as nn

class DenseGATConv(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0):
		super().__init__()
		self.lin = nn.Linear(in_channels, out_channels, bias=False)

		# Wektory uwagi dla węzła źródłowego (src) i docelowego (dst)
		self.att_src = nn.Linear(out_channels, 1, bias=False)
		self.att_dst = nn.Linear(out_channels, 1, bias=False)

		self.leaky_relu = nn.LeakyReLU(0.2)
		self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

	def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
		val = self.lin(x)  # [Batch, Nodes, OutChannels]

		query = self.att_src(val)  # [Batch, Nodes, 1]
		keys = self.att_dst(val)  # [Batch, Nodes, 1]

		e = query + keys.transpose(1, 2)
		e = self.leaky_relu(e)

		# Maskowanie strukturą grafu (Połączenia to 1 [log(1)==0], brak połączenia to 0 [log(0+) =-inf])
		attention = e + torch.log(adj + 1e-6)

		# Softmax na ostatnim wymiarze
		attention = torch.softmax(attention, dim=-1)
		attention = self.dropout(attention)

		# Opcjonalnie mnożenie przez macierz A, aby wyzerować nieistniejące krawędzie i zwiększyć znaczenie macierzy A
		attention = attention * adj
		attention = attention / (attention.sum(dim=-1, keepdim=True) + 1e-6)  # re-normalizacja

		# 5. Agregacja cech od sąsiadów
		out = torch.bmm(attention, val)  # [Batch, Nodes, OutChannels]
		return out

