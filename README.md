# TODO 2
- [ ] Wygenerować lepszy zbiór framsticków (taki, w którym bierzemy osobników z hall of fame dla każdej ?epoki?)
  - Wiele odtworzeń
  - Równomierne rozłożenie przykładów
- [ ] CMA-ES jako r
- [ ] Badać poziom poprawności dekodowania (przyda się przy wersji z douczaniem)
- [ ] Implementacja wersji z douczaniem
  - [ ] 1. W trakcie działania zapisywać genotypy powstające z dekodowania
  - [ ] 2. Jeżeli poziom odtwarzania spadnie poniżej progu, douczyć autoenkoder zapisanymi osobnikami
  - [ ] 3. Po douczeniu uruchamiamy ponownie algorytm ale ze startowego (lub nie) miejsca i wracamy do kroku 1
- [ ] Dokończyć reimplementacje, weryfikując poprawność reimplementacji
- [?] Zaimplementować locality loss dla similarity
- [ ] Slurp, slurm
- [ ] Wypisać co znajdzie się w wynikach, jakie wykresy itp.
- [X] Poprawić stratę locality na spearmana

# TODO
- [X] Wygenerowanie zbioru framsticków (ok 29000 patyczaków). Najlepiej równomierny podział na fitnessie
- [ ] Wybrać recenzenta
- [ ] Odpowiednia konwersja f0 -> X, A
- [ ] Odpowiednia konwersja X' A' -> f0
- [ ] Wykonanie całego pipeline uczenia wraz z zapisywaniem poprawnie wyników
- [ ] Implementacja naprawiania i weryfikowania wyników
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

# Dodatkowe informacje
