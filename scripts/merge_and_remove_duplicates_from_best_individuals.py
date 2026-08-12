import yaml
import os
import glob
import json


def main():
	config_path = "../configs/generate_individual_data_config.yaml"

	# Wczytanie konfiguracji
	try:
		with open(config_path, "r", encoding="utf-8") as f:
			config = yaml.safe_load(f)
	except FileNotFoundError:
		print(f"Nie znaleziono pliku konfiguracyjnego: {config_path}")
		return

	orig_filepath = config.get("result_filepath")
	if not orig_filepath:
		print("Błąd: Brak klucza 'result_filepath' w pliku konfiguracyjnym.")
		return

	# Rozdzielamy na nazwę i rozszerzenie
	base_name, ext = os.path.splitext(orig_filepath)

	# Szukamy wszystkich plików
	search_pattern = f"{base_name}_*{ext}"
	merged_filepath = f"{base_name}_merged{ext}"

	file_list = glob.glob(search_pattern)

	# Upewniamy się, że przez przypadek nie wczytamy pliku "_merged",
	# jeśli skrypt jest uruchamiany po raz kolejny
	file_list = [f for f in file_list if "merged" not in f]

	if not file_list:
		print(f"Nie znaleziono plików pasujących do wzorca: {search_pattern}")
		return

	print(f"Znaleziono {len(file_list)} plików do scalenia. Rozpoczynam przetwarzanie...")

	# Przetwarzanie i usuwanie duplikatów
	seen_genotypes = set()
	total_records = 0

	with open(merged_filepath, "w", encoding="utf-8") as out_f:
		for file_path in file_list:
			with open(file_path, "r", encoding="utf-8") as in_f:
				for line in in_f:
					line = line.strip()
					if not line:
						continue

					total_records += 1
					try:
						record = json.loads(line)
						genotype = record.get("genotype")

						# Jeśli genotyp istnieje i jeszcze go nie widzieliśmy -> zapisujemy
						if genotype and genotype not in seen_genotypes:
							seen_genotypes.add(genotype)
							out_f.write(line + "\n")

					except json.JSONDecodeError:
						print(f"Ostrzeżenie: Błąd parsowania JSON w pliku {file_path}, pomijam linię.")

	# Podsumowanie
	duplicates_removed = total_records - len(seen_genotypes)

	print("\n--- Podsumowanie operacji ---")
	print(f"Przetworzono rekordów:   {total_records}")
	print(f"Unikalnych genotypów:    {len(seen_genotypes)}")
	print(f"Usuniętych duplikatów:   {duplicates_removed}")
	print(f"Zapisano wynik do pliku: {merged_filepath}")


if __name__ == "__main__":
	main()