"""
=============================================================================
MODELLO FCNN - Fully Connected Neural Network (Parametrico)
=============================================================================
Corso: Neural Network Deep Learning - Esame Magistrale

DESCRIZIONE TEORICA:
Questo modulo implementa un Perceptron Multistrato (MLP/FCNN) generico.
Una FCNN è caratterizzata da:
1. Connessioni dense: Ogni neurone del layer L è connesso a tutti i neuroni del layer L-1.
2. Assenza di prior spaziali: L'immagine 2D viene "appiattita" (flattening), 
   perdendo la struttura spaziale dei pixel (vicinanza, località).
3. Universal Approximation Theorem: Con abbastanza neuroni e non-linearità, 
   può approssimare qualsiasi funzione continua (teoricamente).

IMPLEMENTAZIONE PYTORCH:
- Eredita da nn.Module: la classe base per tutti i moduli neurali.
- Gestione dinamica: Permette di variare profondità e larghezza della rete 
  per analizzare l'impatto degli iperparametri (come richiesto dalla traccia).
"""

import torch
import torch.nn as nn

class FCNN(nn.Module):
    """
    Rete Fully Connected parametrica per esperimenti sistematici.
    
    Permette di variare l'architettura (profondità e ampiezza) e le funzioni 
    di attivazione per studiare il comportamento del gradiente e la capacità 
    di generalizzazione.
    
    Args:
        input_size (int): Dimensione del vettore di input (784 per MNIST appiattito).
        hidden_sizes (list): Lista di interi che definisce l'architettura.
                             Es: [256, 128] crea due hidden layer.
        num_classes (int): Dimensione dell'output (10 classi per MNIST).
        activation (str): 'relu', 'sigmoid', 'tanh', 'leaky_relu'.
        dropout (float): Probabilità di azzerare un neurone (regolarizzazione).
    """
    
    def __init__(self, input_size=784, hidden_sizes=[256, 128], num_classes=10, 
                 activation='relu', dropout=0.0):
        # Inizializzazione della classe padre nn.Module.
        # FONDAMENTALE: registra la classe nel grafo computazionale di PyTorch.
        super(FCNN, self).__init__()
        
        self.input_size = input_size
        
        # =================================================================
        # 1. SELEZIONE DELLA FUNZIONE DI ATTIVAZIONE (Non-linearità)
        # =================================================================
        # Senza attivazioni non-lineari, una rete profonda sarebbe matematicamente
        # equivalente a un singolo layer lineare (W2(W1(x)) = W_tot(x)).
        #
        # - ReLU: f(x)=max(0,x). Standard odierno. Mitiga il Vanishing Gradient
        #         perché la derivata è 1 per x>0. Calcolo efficiente.
        # - Sigmoid: f(x)=1/(1+e^-x). Storica. Mappa in (0,1).
        #            PROBLEMA: Satura agli estremi -> gradienti vicini a 0 -> 
        #            la rete non impara (Vanishing Gradient).
        # - Tanh: Simile a Sigmoid ma centrata in 0 (range -1, 1). Spesso converge
        #         meglio della Sigmoid ma soffre dello stesso problema di saturazione.
        # =================================================================
        activation_functions = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.01), # Variante ReLU che non "muore" per x<0
        }
        
        if activation not in activation_functions:
            raise ValueError(f"Attivazione '{activation}' non supportata. "
                           f"Usa: {list(activation_functions.keys())}")
        
        self.activation = activation_functions[activation]
        self.activation_name = activation
        
        # =================================================================
        # 2. COSTRUZIONE DINAMICA DELL'ARCHITETTURA
        # =================================================================
        # Usiamo nn.ModuleList invece di una lista Python standard ([]).
        # PERCHÉ?
        # Se usassimo una lista normale (self.layers = []), PyTorch NON
        # "vederebbe" i layer al suo interno. Di conseguenza, model.parameters()
        # restituirebbe una lista vuota e l'ottimizzatore non aggiornerebbe nulla.
        # nn.ModuleList registra correttamente i sottomoduli.
        # =================================================================
        
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        prev_size = input_size
        
        # Iteriamo sulla lista hidden_sizes per creare la topologia richiesta
        for hidden_size in hidden_sizes:
            # Linear: y = xA^T + b
            self.layers.append(nn.Linear(prev_size, hidden_size))
            
            # Dropout: Tecnica di regolarizzazione. Durante il training, spegne
            # casualmente dei neuroni con probabilità p.
            # Questo costringe la rete a imparare feature ridondanti e robuste,
            # prevenendo l'overfitting (co-adaptation of neurons).
            self.dropouts.append(nn.Dropout(dropout))
            
            prev_size = hidden_size
        
        # Layer di Output: Mappa l'ultimo hidden state alle classi finali (logits)
        self.output_layer = nn.Linear(prev_size, num_classes)
        
        # =================================================================
        # 3. INIZIALIZZAZIONE DEI PESI (Weight Initialization)
        # =================================================================
        # I pesi non possono essere inizializzati tutti a zero (Symmetry Breaking problem).
        # Se fossero zero, tutti i neuroni calcolerebbero lo stesso gradiente 
        # e rimarrebbero identici per sempre.
        # 
        # Usiamo strategie euristiche basate sulla funzione di attivazione scelta.
        # =================================================================
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Applica l'inizializzazione corretta in base alla teoria.
        """
        for layer in self.layers:
            # KAIMING (HE) INITIALIZATION
            # Ottimale per ReLU. Mantiene la varianza dell'output costante rispetto all'input
            # tenendo conto che ReLU azzera metà degli input (i negativi).
            if self.activation_name in ['relu', 'leaky_relu']:
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            
            # XAVIER (GLOROT) INITIALIZATION
            # Ottimale per Sigmoid/Tanh. Mantiene la varianza dei gradiente simile
            # tra forward e backward pass per evitare esplosione/sparizione del gradiente
            # nella zona lineare dell'attivazione.
            else:
                nn.init.xavier_normal_(layer.weight)
            
            # BIAS
            # È buona norma inizializzare i bias a 0.
            nn.init.zeros_(layer.bias)
        
        # Layer di output
        # Si usa solitamente Xavier perché è una trasformazione lineare pura (o verso Softmax)
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x):
        """
        Definisce il flusso dei dati (Forward Pass).
        PyTorch costruisce automaticamente il grafo per il Backward Pass (Autograd).
        
        Args:
            x: Tensore di input (Batch_Size, Canali, Altezza, Larghezza)
        """
        # STEP 1: FLATTENING
        # Le FCNN richiedono vettori 1D. Trasformiamo il tensore (N, 1, 28, 28)
        # in (N, 784). .view() è efficiente perché non copia memoria.
        x = x.view(x.size(0), -1)
        
        # STEP 2: HIDDEN LAYERS LOOP
        # Applichiamo sequenzialmente: Lineare -> Attivazione -> Dropout
        for layer, dropout in zip(self.layers, self.dropouts):
            x = layer(x)           # Trasformazione affine
            x = self.activation(x) # Introduzione non-linearità
            x = dropout(x)         # Regolarizzazione (attiva solo in train mode)
        
        # STEP 3: OUTPUT LAYER
        # NOTA IMPORTANTE: Qui NON applichiamo Softmax.
        # Restituiamo i "Logits" (punteggi grezzi).
        # La funzione di loss `nn.CrossEntropyLoss` di PyTorch applica internamente
        # LogSoftmax + NLLLoss per motivi di stabilità numerica.
        x = self.output_layer(x)
        
        return x
    
    def count_parameters(self):
        """
        Utility per contare i parametri addestrabili (pesi + bias).
        Utile per confrontare la complessità dei modelli nella relazione.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self):
        """Stampa una descrizione leggibile dell'architettura."""
        hidden_str = " -> ".join(str(l.out_features) for l in self.layers)
        return (f"FCNN(In: {self.input_size} -> Hidden: {hidden_str} -> "
                f"Out: {self.output_layer.out_features}, "
                f"Act: {self.activation_name})")