# TODO 2
- [X] Wygenerować lepszy zbiór framsticków (taki, w którym bierzemy osobników z hall of fame dla każdej ?epoki?)
  - Wiele odtworzeń
  - Równomierne rozłożenie przykładów
- [ ] Slurp, slurm
- [ ] Wypisać co znajdzie się w wynikach, jakie wykresy itp.
  - [ ] Wizualizacje ?
- [?] Zaimplementować locality loss dla similarity
  - [ ] Zweryfikować, dlaczego przetwarza się niemiłosiernie długo
# TODO
- [ ] Własna implementacja
  - [ ] Własna warstwa gęsta
  - [ ] Własna warstwa konwolucyjna
  - [ ] Własny enkoder
  - [ ] Własny dekoder
  - [ ] Własny autoenkoder
  - [ ] Własny wariacyjny autoenkoder
  - [ ] Działanie na fenotypach
  - [ ] Zwrócenie uwagi na możliwe warstwy grafowe konwolucyjne
  - [ ] Poprawna weryfikacja przebiegu uczenia autoenkodera
- [ ] Przygotować prezentację z wynikami
  - [ ] Omówienie architektury
    - [ ] Autoenkoder
    - [ ] Dodatkowe człony funkcji straty
  - [ ] Dane uczące do autoenkodera, wykres zbioru uczącego (zakres fitness na liczbę osobników)
  - [ ] Wyniki eksperymentu

## Jakie eksperymenty?
- Bazowy algorytm ewolucyjny (1 bądź 2 rodzaje [GP i GE])
Zwykły i wariacyjny:
    Trzy locality loss: part number, fitness, similarity:
    - Na nauczonym autoenkoderze
    - Z douczaniem
    - Ten z reimplementacji
Łącznie: 1 + 2 * 3 * 3 = 19

> Obliczać również czas i warunkować eksperyment czasem przetwarzania

# Wizualizacje, grafiki, wykresy itp.
1. Zbiór uczący
   2. Tabela z rozkładem wysokości, liczby części
   3. Histogram z rozkładu wysokości i liczby części
   4. Dodatkowe informacje, w tym np. liczba uruchomień, sposób generacji zbioru
2. 

# Dodatkowe informacje
