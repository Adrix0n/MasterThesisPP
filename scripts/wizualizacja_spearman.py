import torch
import matplotlib.pyplot as plt


def compute_soft_ranks(x_dense, x_batch_frozen):
	"""
	Funkcja pomocnicza symulująca to, co dzieje się wewątrz metody CDF.
	x_dense: gęsta siatka wartości do narysowania płynnej linii
	x_batch_frozen: punkty z naszego obecnego batcha (stanowiące szczeble drabinki)
	"""
	N = x_batch_frozen.size(0)

	# Nasza posortowana drabinka z batcha
	x_sorted, _ = torch.sort(x_batch_frozen)

	# Szukamy miejsc dla wszystkich punktów siatki
	idx = torch.searchsorted(x_sorted, x_dense)

	idx_left = torch.clamp(idx - 1, min=0, max=N - 2)
	idx_right = idx_left + 1

	val_left = x_sorted[idx_left]
	val_right = x_sorted[idx_right]

	diff = torch.clamp(val_right - val_left, min=1e-6)

	# Interpolacja
	fraction = (x_dense - val_left) / diff
	fraction = torch.clamp(fraction, 0.0, 1.0)

	# Ekstrapolacja płaska poza zakresem danych (dla czytelności wykresu)
	fraction[x_dense < x_sorted[0]] = 0.0
	fraction[x_dense > x_sorted[-1]] = 1.0

	soft_ranks = idx_left.float() + fraction
	return soft_ranks


# 1. Definiujemy przykładowy batch (wartości wylosowane przez sieć dla 6 elementów)
# Zauważ, że niektóre są blisko siebie (np. 1.0 i 1.2), a inne daleko (5.0)
x_batch = torch.tensor([-2.0, -0.5, 1.0, 1.2, 3.5, 5.0])

# 2. Tworzymy gęstą siatkę punktów (X) przypiętą do grafu obliczeń
x_grid = torch.linspace(-3.0, 6.0, 500, requires_grad=True)

# 3. Przepuszczamy siatkę przez nasz algorytm
soft_ranks = compute_soft_ranks(x_grid, x_batch)

# 4. Obliczamy pochodną (gradient), sumując wyjście, aby odpalić backward()
soft_ranks.sum().backward()
gradients = x_grid.grad.numpy()

# === WIZUALIZACJA ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Wykres 1: Jak wartość mapuje się na rangę
ax1.plot(x_grid.detach().numpy(), soft_ranks.detach().numpy(),
		 label="Różniczkowalna Miękka Ranga (CDF)", color="blue", linewidth=2)
ax1.scatter(x_batch.numpy(), torch.arange(len(x_batch)).numpy(),
			color="red", zorder=5, s=50, label="Rzeczywiste punkty w batchu")

ax1.set_title("Interpolacja Rang (Piecewise Linear CDF)")
ax1.set_ylabel("Przypisana Ranga")
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend()

# Wykres 2: Przepływ gradientu (pochodna)
ax2.plot(x_grid.detach().numpy(), gradients,
		 label="Gradient (Siła sygnału)", color="green", linewidth=2)
ax2.set_title("Gradient funkcji w zależności od odległości między punktami")
ax2.set_xlabel("Wartość wyjściowa neuronu (X)")
ax2.set_ylabel("Pochodna")
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend()

plt.tight_layout()
plt.show()