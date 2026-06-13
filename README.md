# TODO
- [X] Wygenerowanie zbioru framsticków (ok 29000 patyczaków). Najlepiej równomierny podział na fitnessie
- [ ] Wybrać recenzenta
- [ ] Odpowiednia konwersja f0 -> X, A
- [ ] Odpowiednia konwersja X' A' -> f0
- [ ] Utworzenie wersji z przetwarzaniem zamiennym (autoenkoder -> optymalizacja -> autoenkoder -> ...) 
- [ ] Implementacja locality loss pod 3 postaciami oddzielnie (oddzielne warianty przetwarzania)
  - [ ] Part number
  - [ ] Fitness value
  - [ ] Dissimilarity (trudne, bo jak identyczna wieża stojąca i leżąca od różny fitness, ale te dissimilarity równe 0)
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
  - [ ] Dane uczące do autoenkoder, wykres zbioru uczącego (zakres fitness na liczbę osobników)
  - [ ] Wyniki eksperymentu


## Jakie eksperymenty?
1. Bazowy algorytm ewolucyjny (1 bądź 2 rodzaje [GP i GE])
2. Cykliczny z przejściami (douczanie <-> optymalizacja przestrzeni) z uwzględnieniem stagnacji
   1. Warianty z part number locality loss
   2. Wariant z fitness value locality loss
   3. Wariant z dissimilarity locality loss
3. Na nauczonym autoenkoderze
   1. Warianty z part number locality loss
   2. Wariant z fitness value locality loss
   3. Wariant z dissimilarity locality loss
4. Wariacyjny cykliczny z przejściami
   1. Tak samo warianty z locality loss
5. Na nauczonym wariacyjnym autoenkoderze
   1. Tak samo warianty z locality loss
6. Ten z reimplementacji (zwykły i wariacyjny):
   1. Warianty z locality loss