---
title: Simulazione degli N-corpi
author: Diamadis Odysseas, Sebastiani William
---

# Il progetto

Il progetto consiste in una simulazione di N corpi per ciascuna delle seguenti modalità
- Algoritmo esaustivo multithread sulla macchina host (OpenMP)
- Algoritmo di Barnes-Hut multithread con OpenMP
- Algoritmo esaustivo con cuda
- Algoritmo di Barnes-Hut con cuda


# Esaustivo con OpenMP

L'algoritmo esaustivo è molto semplice da parallelizzare. È possibile infatti scomporre la computazione
in due fasi principali che non comportano alcuna concorrenza da parte dei thread: la fase di calcolo delle interazioni,
in cui per ciascun corpo viene compilato il vettore delle forze risultanti applicate su di esso, e la fase di applicazione
delle suddette forze.

È sufficiente spezzare queste due fasi in due costrutti
```c
#pragma omp parallel for
```
per ottenere una sufficiente parallelizzazione.

\pagebreak

## Speed up

| thread | tempo (s) | speedup |
|---|---|---|
| 1  | 8288 | 1.00 | 
| 2  | 4199 |0.50|
| 3  | 2864 |0.34|
| 4  | 2175 |0.26|
| 5  | 1800 |0.22|
| 6  | 1911 |0.20|
| 7  | 1651 |0.19|
| 8  | 1496 |0.18|
| 9  | 1384 |0.16|
| 10  | 1274 |0.15|
| 11  | 1190 |0.14|
| 12  | 1137 |0.13|



# Barnes-Hut con OpenMP

L'algoritmo di Barnes-Hut è caratterizzato dalla suddivisione dello spazio in ottanti, i quali costituiranno un albero, detto l'**octree** alle cui foglie saranno presenti i corpi iniziali.
Dei nodi interni viene tenuto conto dei centri di massa dei corpi o dei nodi sottostanti.

In questo modo, quando un corpo è sufficientemente lontano da un gruppo di corpi, questi possono essere trattati come un unico corpo che ha come valori di posizione e di massa quelli del centro di massa del gruppo. Questo riduce la complessità temporale ad un O(logn), contrariamente alla versione esaustiva che richiede O(n^2).

\pagebreak

Il calcolo della forza avviene nel seguente modo

```
Per ciascun corpo C:
   Esegui una visita in profondità dell'albero
   Per ciascun nodo interno:
      se il rapporto tra la distanza tra il corpo ed il centro di massa del nodo corrente e la lunghezza del lato dell'ottante corrente è minore di un certo valore THETA:
            calcola l'interazione tra il corpo ed il centro di massa del nodo
      altrimenti:
            continua a scendere nell'albero
```

## Speed up
Di seguito la tabella con il tempo impiegato in una simulazione.
Il tempo è calcolato come la media dei tempi impiegati in 1000 simulazioni con 5000 corpi casuali.

| thread | tempo (s) | speedup |
|---|---|---|
| 1  | 166.96 | 1.00 | 
| 2  | 110.73 |0.61|
| 3  | 94.79 |0.57|
| 4  | 85.34 |0.51|
| 5  | 80.05 |0.47|
| 6  | 76.28 |0.45|
| 7  | 71.76 |0.42|
| 8  | 72.63 |0.41|
| 9  | 70.05 |0.42|
| 10  | 65.48 |0.40|
| 11  | 64.36 |0.37|
| 12  | 66.72 |0.38|

