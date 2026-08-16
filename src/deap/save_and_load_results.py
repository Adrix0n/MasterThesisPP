import json
import os

def save_genotypes_json(filename, hof):
    # Jeśli plik istnieje – wczytaj istniejące dane
    if os.path.exists(filename):
        with open(filename, "r") as infile:
            try:
                data = json.load(infile)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Aktualny indeks startowy
    start_id = len(data)

    # Dopisz nowe rekordy
    for idx, ind in enumerate(hof):
        entry = {
            "id": start_id + idx,
            "fitness": list(ind.fitness.values),
            "genotype": getattr(ind, "genotype", ""),
            "latent_vector": list(ind)
        }
        data.append(entry)

    # Zapisz cały plik ponownie
    with open(filename, "w") as outfile:
        json.dump(data, outfile, indent=4)

    print(f"Saved '{filename}' ({len(hof)})")