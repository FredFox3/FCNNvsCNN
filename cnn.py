"""
=============================================================================
MODELLO CNN - Convolutional Neural Network (Parametrico)
=============================================================================
Corso: Neural Network Deep Learning - Esame Magistrale

DESCRIZIONE TEORICA:
A differenza delle FCNN, le CNN sfruttano la struttura spaziale dei dati (immagini).
Si basano su tre concetti chiave (Inductive Bias):
1. Local Connectivity: I neuroni sono connessi solo a una piccola regione dell'input.
2. Parameter Sharing: Lo stesso filtro (kernel) viene usato su tutta l'immagine. 
   Questo riduce drasticamente i parametri e garantisce l'equivarianza alla traslazione.
3. Pooling: Riduce la dimensionalità e introduce invarianza a piccole traslazioni.

IMPLEMENTAZIONE PYTORCH:
Il modello è costruito dinamicamente per supportare le variazioni richieste dalla traccia:
- Numero di filtri (Depth scaling)
- Dimensione del kernel (Receptive field)
- Numero di blocchi (Profondità della rete)
"""

import torch
import torch.nn as nn
import math

class CNN(nn.Module):
    """
    CNN parametrica per l'analisi sistematica degli iperparametri.
    
    Architettura generale:
    Input -> [Conv Block] x N -> Flatten -> Dense -> Output
    
    Dove un [Conv Block] è tipicamente: Conv2d -> BatchNorm -> ReLU -> MaxPool2d
    
    Args:
        num_conv_blocks (int): Numero di stadi di estrazione feature (2 o 3).
                               Determina quanto l'immagine viene rimpicciolita.
        filters (list): Numero di feature maps per ogni blocco.
                        Es: [32, 64] significa che il primo livello estrae 32 pattern,
                        il secondo combina quei 32 in 64 pattern più complessi.
        kernel_size (int): Dimensione spaziale del filtro (3 o 5).
                           Influenza il "Campo Recettivo" (quanto vede il neurone).
        fc_hidden (int): Dimensione del layer denso intermedio.
        dropout (float): Regolarizzazione per prevenire overfitting.
        use_batch_norm (bool): Se True, normalizza le attivazioni interne.
    """
    
    def __init__(self, num_conv_blocks=2, filters=None, kernel_size=3,
                 fc_hidden=128, num_classes=10, dropout=0.0, 
                 use_batch_norm=False):
        super(CNN, self).__init__()
        
        # Gestione parametri di default
        if filters is None:
            filters = [32, 64] if num_conv_blocks == 2 else [16, 32, 64]
        
        # Validazione input per evitare errori criptici a runtime
        assert len(filters) == num_conv_blocks, \
            f"Errore: hai richiesto {num_conv_blocks} blocchi ma hai fornito {len(filters)} dimensioni filtri."
        assert kernel_size in [3, 5], "Kernel size supportati per MNIST: 3 o 5."
        
        self.num_conv_blocks = num_conv_blocks
        self.filters = filters
        self.kernel_size = kernel_size
        
        # =================================================================
        # 1. GESTIONE DEL PADDING E DIMENSIONI
        # =================================================================
        # Vogliamo controllare la riduzione dimensionale SOLO tramite il Pooling.
        # Pertanto, usiamo "Same Padding" nella convoluzione.
        # Formula output size: O = (W - K + 2P)/S + 1
        # Se Stride (S)=1, per avere O=W serve P = (K-1)/2.
        # 
        # - Kernel 3x3 -> Padding 1
        # - Kernel 5x5 -> Padding 2
        # =================================================================
        padding = (kernel_size - 1) // 2
        
        # =================================================================
        # 2. COSTRUZIONE DEI BLOCCHI (FEATURE EXTRACTOR)
        # =================================================================
        # Usiamo nn.ModuleList per registrare i blocchi dinamicamente.
        self.conv_blocks = nn.ModuleList()
        
        in_channels = 1  # MNIST è in scala di grigi (1 canale)
        
        for i, out_channels in enumerate(filters):
            # nn.Sequential permette di impacchettare operazioni in ordine
            block = nn.Sequential()
            
            # A. Convoluzione: Estrazione features lineari
            block.add_module(f'conv{i+1}', nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=not use_batch_norm # Se c'è BN, il bias è ridondante (lo gestisce BN)
            ))
            
            # B. Batch Normalization (Opzionale)
            # Normalizza l'output della convoluzione (media 0, var 1).
            # Riduce l'Internal Covariate Shift e permette learning rate più alti.
            if use_batch_norm:
                block.add_module(f'bn{i+1}', nn.BatchNorm2d(out_channels))
            
            # C. Attivazione: Non-linearità
            block.add_module(f'relu{i+1}', nn.ReLU())
            
            # D. Pooling: Downsampling
            # MaxPool2d(2, 2) dimezza altezza e larghezza.
            # Rende la rete invariante a piccole traslazioni.
            block.add_module(f'pool{i+1}', nn.MaxPool2d(kernel_size=2, stride=2))
            
            # E. Dropout Spaziale (Opzionale)
            # Dropout2d azzera interi canali (feature maps), non singoli pixel.
            if dropout > 0:
                block.add_module(f'dropout{i+1}', nn.Dropout2d(dropout))
            
            self.conv_blocks.append(block)
            in_channels = out_channels # L'output di oggi è l'input di domani
        
        # =================================================================
        # 3. CALCOLO DELLA DIMENSIONE DI FLATTEN
        # =================================================================
        # Per collegare la parte convoluzionale (3D: C x H x W) alla parte 
        # fully connected (1D), dobbiamo sapere quanti neuroni escono.
        #
        # MNIST parte da 28x28. Ogni MaxPool dimezza il lato.
        # - 1 blocco: 28 -> 14
        # - 2 blocchi: 28 -> 14 -> 7
        # - 3 blocchi: 28 -> 14 -> 7 -> 3 (divisione intera)
        # =================================================================
        final_hw = 28 // (2 ** num_conv_blocks)
        
        # Dimensione totale = Numero Canali Finali * Altezza * Larghezza
        self.flatten_size = filters[-1] * final_hw * final_hw
        
        # =================================================================
        # 4. CLASSIFICATORE (DENSE LAYERS)
        # =================================================================
        self.classifier = nn.Sequential(
            nn.Flatten(),                          # Da (N, C, H, W) a (N, C*H*W)
            nn.Linear(self.flatten_size, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(fc_hidden, num_classes)      # Output: 10 classi (Logits)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Inizializzazione pesi 'Kaiming He'.
        Cruciale per reti profonde con ReLU per evitare che il segnale si 
        spenga (vanishing) o esploda nei primi passaggi.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # mode='fan_out' preserva la varianza nel passaggio backward
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass: Immagine -> Features -> Classificazione
        """
        # Fase di Feature Extraction (apprende COSA c'è nell'immagine)
        for block in self.conv_blocks:
            x = block(x)
        
        # Fase di Classificazione (decide CHE NUMERO è basandosi sulle features)
        x = self.classifier(x)
        
        return x # Ritorna Logits (non probabilità), la Loss gestirà il Softmax
    
    def count_parameters(self):
        """Conta i parametri addestrabili per confronto con FCNN."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self):
        """Stampa diagnostica dell'architettura."""
        filters_str = " -> ".join(map(str, self.filters))
        return (f"CNN(Blocks: {self.num_conv_blocks}, "
                f"Filters chain: {filters_str}, "
                f"Kernel: {self.kernel_size}x{self.kernel_size}, "
                f"Flatten dim: {self.flatten_size}, "
                f"Params: {self.count_parameters():,})")