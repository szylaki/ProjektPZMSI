# ProjektPZMSI
Sebastian Śliwa, Agnieszka Rutkowska, Igor Szylak, Alona Skyba

# Opis TensorFlow i Keras
TensorFlow jest platformą przeznaczoną do tworzenia i uczenia sieci neuronowych. Został stworzony z myślą o prostym wykorzystaniu w praktyce, posiada wbudowane API Keras, dzięki któremu użytkownik ma ułatwiony dostęp do funkcjonalności TensorFlow. 
Jego twórcą jest Google Brain Team. Sama biblioteka składa się z kilku modułów. W najniższej jego warstwie został zaimplementowany rozproszony silnik wykonawczy, napisany w języku C++. Nad tą warstwą znajdują się różne frontendy napisane m.in. w Pythonie i C++. Nad tą warstwą jest pierwsza warstwa API, która upraszcza interfejs dla modeli głębokiego uczenia. Finalnie, na szczycie Tensora jest wysokoabstrakcyjne API, takie jak Keras albo Estimator API. Biblioteka TensorFlow jest opensource.
# Zastosowanie w projekcie
W naszym projekcie wykorzystaliśmy bibliotekę TensorFlow do systemu rozpoznawania twarzy. Nauczamy sieć za pomocą danych OlivettiFaces z pakietu Sci-kit learn, a następnie wybieramy jakieś nowe losowe zdjęcie i staramy się je dopasować do wybranej osoby.
