# Confronto Sistematico: FCNN vs CNN su MNIST

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

> **Progetto per il corso di Neural Network Deep Learning**  
> **Laurea Magistrale in Informatica** - Università degli Studi di Napoli Federico II  
> **Anno Accademico:** 2025/2026  
> **Autori:** Alfredo Volpe, Angelo Paolella

---

## 📋 Descrizione del Progetto

Questo repository ospita un "laboratorio virtuale" progettato per confrontare scientificamente due paradigmi di Deep Learning fondamentali: **Fully Connected Neural Networks (FCNN)** e **Convolutional Neural Networks (CNN)**.

Utilizzando il dataset **MNIST** come benchmark, il progetto analizza come diverse scelte architetturali influenzino:
1.  **Accuratezza di classificazione** (Generalizzazione).
2.  **Efficienza parametrica** (Numero di parametri vs Performance).
3.  **Dinamiche di apprendimento** (Velocità di convergenza, Vanishing Gradient).

Il confronto segue il principio *Ceteris Paribus*: tutti gli iperparametri di controllo (Learning Rate, Ottimizzatore, Seed, Batch Size) sono mantenuti costanti per isolare l'impatto delle modifiche architetturali.

---

## 🧠 Architetture Analizzate

Il codice è modulare e permette la generazione dinamica delle reti.

### 1. FCNN (Fully Connected Neural Network)
Implementata in `fcnn.py`.
*   **Caratteristiche:** Input "appiattito" (Flattening 2D -> 1D), connessioni dense.
*   **Variabili di Studio:**
    *   **Profondità:** Shallow (1 layer), Baseline (2 layers), Deep (3 layers).
    *   **Ampiezza:** Bottle-neck vs High Capacity.
    *   **Attivazioni:** ReLU, Sigmoide (analisi vanishing gradient), Tanh.

### 2. CNN (Convolutional Neural Network)
Implementata in `cnn.py`.
*   **Caratteristiche:** Sfruttamento della struttura spaziale, condivisione dei pesi, invarianza alla traslazione (Pooling).
*   **Variabili di Studio:**
    *   **Filtri (Depth Scaling):** Bassa, Media, Alta capacità.
    *   **Kernel Size (Receptive Field):** 3x3 vs 5x5.
    *   **Profondità:** 2 vs 3 blocchi convoluzionali.

---

## 🔬 Metodologia Sperimentale

Per garantire la riproducibilità e l'equità del confronto, sono stati fissati i seguenti parametri di controllo nel file `confronto_fcnn_cnn.py`:

| Parametro | Valore | Note |
| :--- | :--- | :--- |
| **Ottimizzatore** | Adam | Gestione adattiva del LR |
| **Learning Rate** | 0.001 | Standard per convergenza stabile |
| **Batch Size** | 256 | Compromesso stabilità/velocità |
| **Early Stopping** | Patience=5 | Prevenzione overfitting |
| **Random Seed** | 42 | Riproducibilità deterministica |
| **Weight Init** | Kaiming He | Ottimizzato per ReLU |

---

## 📊 Risultati Chiave

Dall'analisi dei risultati (disponibile nei grafici generati e nel report PDF), emergono le seguenti conclusioni:

1.  **Superiorità delle CNN:** Le CNN superano costantemente le FCNN, raggiungendo un'accuratezza >99% contro il ~98% delle migliori FCNN.
2.  **Efficienza:** Le CNN ottengono risultati migliori con **molti meno parametri**.
    *   *Esempio:* Una CNN ottimizzata (~500k parametri) batte una FCNN ad alta capacità (~800k parametri).
3.  **Dimensione del Kernel:** L'uso di kernel **5x5** (campo recettivo più ampio) ha prodotto il risultato migliore assoluto (**99.21%**).
4.  **Problema del Vanishing Gradient:** Le FCNN con attivazione **Sigmoide** convergono molto più lentamente rispetto a quelle con **ReLU**.

### Grafico Efficienza: Parametri vs Accuracy
*(Esempio concettuale basato sui dati del progetto)*
*   🟢 **CNN (Verde):** Alta accuracy, basso numero di parametri.
*   🔵 **FCNN (Blu):** Accuracy inferiore, alto numero di parametri.

---

## 🚀 Installazione e Utilizzo

### Prerequisiti
Assicurati di avere Python installato. Le dipendenze principali sono `torch`, `torchvision` e `matplotlib`.

1.  **Clona il repository:**
    ```bash
    git clone https://github.com/tuo-username/nome-repo.git
    cd nome-repo
    ```

2.  **Installa le dipendenze:**
    ```bash
    pip install torch torchvision matplotlib
    ```

3.  **Esegui gli esperimenti:**
    Per avviare la suite completa di training e confronto:
    ```bash
    python confronto_fcnn_cnn.py
    ```

    *Nota: Impostare `TEST_MODE = True` nel file `confronto_fcnn_cnn.py` per un debug rapido su poche epoche.*

---

## 📂 Struttura dei File

```text
.
├── cnn.py                  # Definizione classe modello Convolutional Neural Network
├── fcnn.py                 # Definizione classe modello Fully Connected Network
├── confronto_fcnn_cnn.py   # Script principale: training, validazione e plotting
├── experiment_results.json # Log dei risultati (generato post-esecuzione)
├── *.png                   # Grafici di confronto (generati post-esecuzione)
└── README.md               # Documentazione
```

## 🎓 Riferimenti

Il progetto fa riferimento ai concetti trattati nel corso di Neural Network Deep Learning, in particolare:
*   *Inductive Bias* delle CNN (Località, Sharing, Invarianza).
*   *Universal Approximation Theorem* per le FCNN.
*   Tecniche di regolarizzazione (Dropout, Batch Norm, Early Stopping).

---
*© 2026 Alfredo Volpe, Angelo Paolella - Università degli Studi di Napoli Federico II*
