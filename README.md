# TODO 4 praca
- [X] Przygotować diagramy warstw gęstej i konwolucyjnej
- [X] Przygotować diagramy enkodera, dekodera A i dekodera X
- [X] Przetłumaczenie rzdziału 3
- [ ] Usunięcie appendixów
- [ ] Przygotowanie rozdziału 4
- [ ] Przygotwanie  skryptu do obsługi wyników
- [ ] Przygotwanie wyników, dopisanie, przetłuaczenie i fajrant

# TODO 3 (ostatnie)
- [X] CMA-ES
  - [X] Zastosować techniki douczania:
    - [X] Kolejke odrzucającą najstarsze osobniki, utrzymująca stały rozmiar 
    - [X] Z dodawaniem osobników do zbioru
    - [X] Kolejka odrzucająca najsłabsze osobniki
  - [X] w CMA_ES nie naprawiać osobników, żeby optymalizacja nie chciała dążyć coraz bardziej w zepsute rejiny
- [X] Poprawić ostatecznie implementacje
  - [X] Warstwa konwolucyjna grafowa
  - [X] Zastanowić się, co właściwie robi dywergencja Kullbacka_Leiblera
    - Bada, czy rozkład wartości przestrzeni ukrytej Z rzeczywiście zgadza się z zadanym rozkładem normalnym (wektor np. 15 wartości, wartości te mają rozkład bliski rozkładowi normalnemu)
- [X] Przygotować skrypt/notebook do automatycznego przeprowadzenia wybranych eksperymentów (potrzebne do slurma)
- [ ] Przygotować notebooki z wizualizacją odpowiednich wyników
  - [ ] Wizualizacje uczenia
  - [ ] Wykresy porównawcze kombinacji autoenkodera
  - [ ] Wykresy podobne do tych z Klejdy

# TODO
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
      Trzy rozmiary latent space:
        Dwa różne zestawy warstw:
          - Na nauczonym autoenkoderze,
          - Z douczaniem,
          - Ten z reimplementacji

Łącznie: 1 + 2 * 3 * 3 * 3 * 2 = 109

> Obliczać również czas i warunkować eksperyment czasem przetwarzania

Eksperyment encode i decode w wielu powtórkach. Jeżeli jest strata, to możliwe, że zdekodowany osobnik poddany ponownie kodowaniu i dekodowaniu będzie jeszcze bardziej zniszczony 

Eksperyment różną liczbą zbioru uczącego (Wygląda na to, że większa liczba korzystnie wpływa na jakość rozwiązań)


# Wizualizacje, grafiki, wykresy itp.
1. Zbiór uczący
    1. Tabela z rozkładem wysokości, liczby węzłów, liczby połączeń (std, mean, min, max)
    2. Histogram z rozkładu wysokości, liczby części i liczby połączeń
    3. Dodatkowe informacje, w tym np. liczba uruchomień, sposób generacji zbioru
2. Trening
    1. Porównanie loss względem różnego rozmiaru latent space
    2. Porównanie straty X oraz A dla wersji zwykłej oraz wariacyjnej
    3. Różnice na stracie względem różnej liczby warstw
    4. Różnice na stracie względem różnego locality loss, w tym None
3. Przebieg ewolucji
    1. Porównanie przebiegu ewolucji EA na zwykłej i zmodyfikowanej przestrzenii
       - Czas
       - Wykres najlepszego
       - Wykres średnich
    2. Porównanie różnych najlepszych uzyskanych modeli względem bazowej ewolucji
     

wyniki z pracy:
- porównanie loss dla różnej liczby warstw
- przebieg straty rekonstrukcji dla całości, X, A
- Średnia strata dla 3 różnych wielkości latent space oraz czterech różnych warstw Conv
- Wpływ mutacji na losowych przykładach w GAE i VGAE w zależności od siły mutacji
- Dystans do nie gorszego osobnika w przestrzeni ukrytej dla GAE i VGAE
- Strata dla różnych locality loss, w tym również None
- Przebiegi ewolucji w algorytmie działającym w przestrzeni ukrytej
- Porównanie 5 najlepszych modeli z bazowym: baseline VGAE_parts, VGAE_sim, VGAE_sim, VGAE_parts, VGAE_none, GAE_none, GAE_parts, GAE_sim, GAE_fit


# Dodatkowe informacje
external: https://drive.google.com/file/d/1ko3txj3QBHBuU5Gpo-HfEzKoxP2eDdRl/view?usp=sharing