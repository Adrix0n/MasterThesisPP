import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

MAX_PARALLEL = 10  # Ile procesów na raz lokalnie (np. 1 lub 2)
CONFIG_FILE = "../configs/experiments_mini.txt"


def run_experiment(command_args: str):
	cmd = f"python ../notebooks/slurm_model_training.py {command_args}"
	print(f"[START] {cmd}")
	result = subprocess.run(cmd, shell=True)
	if result.returncode != 0:
		print(f"[ERROR] Eksperyment zakończony błędem: {cmd}")
	else:
		print(f"[DONE] {cmd}")


if __name__ == "__main__":
	with open(CONFIG_FILE, "r", encoding="utf-8") as f:
		tasks = [line.strip() for line in f if line.strip() and not line.startswith("#")]

	print(f"Znaleziono {len(tasks)} zadań. Start (max {MAX_PARALLEL} na raz)...")

	with ProcessPoolExecutor(max_workers=MAX_PARALLEL) as executor:
		executor.map(run_experiment, tasks)