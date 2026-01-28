# CIFAR-10 Animals vs Vehicles – CNN Classification

## Opis projektu
Projekt przedstawia zastosowanie konwolucyjnych sieci neuronowych (CNN) do binarnej klasyfikacji obrazów ze zbioru CIFAR-10. Oryginalny problem dziesięcioklasowy został uproszczony do dwóch klas: zwierzęta oraz pojazdy. Celem projektu jest analiza wpływu liczby warstw splotowych na jakość klasyfikacji.

---

## Zbiór danych
W projekcie wykorzystano zbiór CIFAR-10, który zawiera kolorowe obrazy o rozmiarze 32×32 piksele, przypisane do 10 klas.

Podział klas:
- Zwierzęta: bird, cat, deer, dog, frog, horse  
- Pojazdy: airplane, automobile, ship, truck  

Etykiety zostały przekształcone do postaci binarnej:
- 0 – zwierzę  
- 1 – pojazd  

---

## Architektura modeli
Zaprojektowano i porównano trzy modele CNN różniące się liczbą warstw splotowych:
- Model 1 – jedna warstwa splotowa  
- Model 2 – dwie warstwy splotowe  
- Model 3 – trzy warstwy splotowe  

Każdy model zawiera:
- warstwy Conv2D z funkcją aktywacji ReLU  
- warstwy MaxPooling2D  
- warstwę Flatten  
- warstwy Dense  
- warstwę wyjściową z funkcją Softmax  

---

## Proces uczenia
- Podział danych: 30% trening / 70% test  
- Liczba epok: 10  
- Rozmiar batcha: 64  
- Optymalizator: Adam  
- Funkcja straty: categorical_crossentropy  
- Metryka: accuracy  

Uczenie realizowane jest z wykorzystaniem propagacji wstecznej błędu.
