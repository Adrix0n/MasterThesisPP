import sys
sys.path.append('../')
import torch
from utils.FramsticksGraphDataset import FramsticksGraphDataset
from utils.FramsticksPostProcessor import FramsticksPostProcessor
torch.set_printoptions(precision=10)
example_f0_individual = "//0\np:1.4554266080933014, 0.3524548501493414, -0.6360992981341942, fr=0.343, ing=0.251\np:0.2670381981844318, -0.2739245553662239, -1.0595551083384938, fr=0.842, ing=0.267\np:0.34743911607089006, -0.19253797145881857, 0.9369698617940858, fr=0.282, ing=0.451\np:1.4209797252347984, -0.7827931483818612, -0.6436342716281735, fr=0.527, ing=0.211\np:-0.20238053000679912, -0.3548171635176762, 2.8528414286303727, ing=0.485\nj:2, 4, rotstif=0.769\nj:3, 2, rotstif=0.965\nj:0, 2, stif=0.994, rotstif=0.967\nj:1, 2, rotstif=0.952\n"

print(f"Osobnik przed modyfikacją: {example_f0_individual}")

X_matrix, a_matrix, parts_num = FramsticksGraphDataset.parse_f0_to_matrices(example_f0_individual, 25)

print(f"macierz x: {X_matrix}")
print(f"macierz a: {a_matrix}")
print(f"Liczba elementów: {parts_num}")


parser = FramsticksPostProcessor()
_, parsed_to_f0, _ , tries, _ = parser.process(X_matrix,a_matrix)

print(f"Osobnik przed modyfikacją: {example_f0_individual}")
print(f"Osobnik po modyfikacji: {parsed_to_f0}")
print(f"Suma prób: {tries}")
