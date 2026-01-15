"""
=============================================================================
CONFRONTO SISTEMATICO FCNN vs CNN su MNIST (Training & Analysis Script)
=============================================================================
Corso: Neural Network Deep Learning - Esame Magistrale

DESCRIZIONE METODOLOGICA:
Questo script funge da "Laboratorio Virtuale". Segue il metodo scientifico
per confrontare due paradigmi di Deep Learning (Fully Connected vs Convolutional).

PRINCIPI DI IMPLEMENTAZIONE:
1. Ceteris Paribus (A parità di altre condizioni):
   Per un confronto onesto, entrambi i modelli usano lo stesso ottimizzatore (Adam),
   lo stesso Learning Rate, stessa Batch Size e stessi dati di training/validation.
   
2. Riproducibilità:
   Viene fissato un Random Seed (42) per garantire che i pesi iniziali e lo 
   split dei dati siano identici ad ogni esecuzione.

3. Prevenzione dell'Overfitting:
   Implementazione di Early Stopping basato sulla validation accuracy.
   Si salva il modello solo quando la validation metric migliora significativamente.

4. Modularità:
   Le architetture sono importate da file esterni per mantenere il codice pulito
   e separare la definizione del modello dalla logica di training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
import sys
from importlib import import_module

# =============================================================================
# CONFIGURAZIONE ESPERIMENTI
# =============================================================================

# -----------------------------------------------------------------------------
# FLAG DI CONTROLLO
# -----------------------------------------------------------------------------
# TEST_MODE = True: Esegue solo un subset di esperimenti per debug veloce.
# TEST_MODE = False: Esegue la suite completa per la relazione finale.
TEST_MODE = False

# -----------------------------------------------------------------------------
# IPERPARAMETRI COMUNI (Control Variables)
# -----------------------------------------------------------------------------
# Questi parametri rimangono costanti per isolare l'effetto delle modifiche
# architetturali (Variabili Indipendenti).
COMMON_CONFIG = {
    'batch_size': 256,      # Compromesso tra velocità (GPU) e stabilità del gradiente
    'learning_rate': 0.001, # LR standard per Adam. Più alto di SGD perché Adam è adattivo.
    'max_epochs': 50,       # Tetto massimo se la rete non converge prima
    'patience': 5,          # Early Stopping: stop se non migliora per 5 epoche
    'min_delta': 0.15,      # Soglia di "miglioramento significativo" (evita rumore statistico)
    'num_workers': 4,       # Parallelizzazione caricamento dati CPU -> GPU
    'seed': 42              # Seme per RNG (Random Number Generator)
}

# -----------------------------------------------------------------
# PIANO DEGLI ESPERIMENTI FCNN
# -----------------------------------------------------------------
# Obiettivo: Analizzare l'impatto di Profondità (Layers), Ampiezza (Neurons)
# e Non-linearità (Activations) su un'architettura densa.

FCNN_EXPERIMENTS = {
    # GRUPPO A: Variazione Profondità (Vanishing Gradient vs Capacity)
    'fcnn_1layer_128': {
        'hidden_sizes': [128],
        'activation': 'relu',
        'description': '1 hidden layer (128) - Shallow Network',
        'test_priority': True 
    },
    'fcnn_2layer_256_128': {
        'hidden_sizes': [256, 128],
        'activation': 'relu',
        'description': '2 hidden layers (256→128) - Baseline',
        'test_priority': False
    },
    'fcnn_3layer_512_256_128': {
        'hidden_sizes': [512, 256, 128],
        'activation': 'relu',
        'description': '3 hidden layers - Deep Network',
        'test_priority': False
    },
    
    # GRUPPO B: Variazione Ampiezza (Underfitting vs Overfitting)
    'fcnn_2layer_64_32': {
        'hidden_sizes': [64, 32],
        'activation': 'relu',
        'description': '2 layers piccoli (Bottle-neck risk)',
        'test_priority': False
    },
    'fcnn_2layer_512_256': {
        'hidden_sizes': [512, 256],
        'activation': 'relu',
        'description': '2 layers grandi (High Capacity)',
        'test_priority': False
    },
    
    # GRUPPO C: Funzioni di Attivazione (Dinamica del gradiente)
    'fcnn_2layer_relu': {
        'hidden_sizes': [256, 128],
        'activation': 'relu',
        'description': 'ReLU (Standard)',
        'test_priority': False
    },
    'fcnn_2layer_tanh': {
        'hidden_sizes': [256, 128],
        'activation': 'tanh',
        'description': 'Tanh (Zero-centered but saturating)',
        'test_priority': False
    },
    'fcnn_2layer_sigmoid': {
        'hidden_sizes': [256, 128],
        'activation': 'sigmoid',
        'description': 'Sigmoid (Vanishing Gradient Risk)',
        'test_priority': True
    },
}

# -----------------------------------------------------------------
# PIANO DEGLI ESPERIMENTI CNN
# -----------------------------------------------------------------
# Obiettivo: Analizzare Feature Extraction, Campo Recettivo e Gerarchia.

CNN_EXPERIMENTS = {
    # GRUPPO A: Numero di Filtri (Feature Richness)
    'cnn_filters_small': {
        'num_conv_blocks': 2,
        'filters': [16, 32],
        'kernel_size': 3,
        'description': 'Low Capacity (16→32)',
        'test_priority': False
    },
    'cnn_filters_medium': {
        'num_conv_blocks': 2,
        'filters': [32, 64],
        'kernel_size': 3,
        'description': 'Medium Capacity (32→64) - Baseline',
        'test_priority': True
    },
    'cnn_filters_large': {
        'num_conv_blocks': 2,
        'filters': [64, 128],
        'kernel_size': 3,
        'description': 'High Capacity (64→128)',
        'test_priority': False
    },
    
    # GRUPPO B: Kernel Size (Receptive Field)
    # Nota: il caso 3x3 è coperto da 'cnn_filters_medium'
    'cnn_kernel_5x5': {
        'num_conv_blocks': 2,
        'filters': [32, 64],
        'kernel_size': 5,
        'description': 'Kernel 5×5 (Larger receptive field)',
        'test_priority': False
    },
    
    # GRUPPO C: Profondità (Spatial Reduction hierarchy)
    # Nota: il caso 2 blocchi è coperto da 'cnn_filters_medium'
    'cnn_3blocks': {
        'num_conv_blocks': 3,
        'filters': [16, 32, 64],
        'kernel_size': 3,
        'description': '3 blocchi (Output spaziale 3x3)',
        'test_priority': True
    },
}


# =============================================================================
# FUNZIONI DI TRAINING (CORE LOGIC)
# =============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Esegue il loop di training per una singola epoca.
    
    Fasi:
    1. Forward pass: calcolo output
    2. Loss computation: calcolo errore
    3. Backward pass: calcolo gradiente (Autograd)
    4. Optimizer step: aggiornamento pesi
    """
    model.train() # Abilita Dropout e Batch Normalization (se presenti)
    total_loss, correct, total = 0.0, 0, 0
    
    for images, labels in train_loader:
        # Sposta i dati sulla GPU se disponibile
        images, labels = images.to(device), labels.to(device)
        
        # 1. Reset gradienti (PyTorch li accumula per default)
        optimizer.zero_grad()
        
        # 2. Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 3. Backward pass
        loss.backward()
        
        # 4. Aggiornamento pesi
        optimizer.step()
        
        # Raccolta statistiche
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1) # Indice con probabilità max
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return total_loss / len(train_loader), 100 * correct / total


def validate(model, val_loader, criterion, device):
    """
    Valuta il modello su dati non visti (Validation o Test).
    Fondamentale usare torch.no_grad() per risparmiare memoria e calcoli.
    """
    model.eval() # Disabilita Dropout e mette BN in modalità inferenza
    total_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad(): # Disabilita il calcolo del gradiente
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return total_loss / len(val_loader), 100 * correct / total


def run_experiment(model, train_loader, val_loader, test_loader, device, 
                   config, experiment_name, verbose=True):
    """
    Gestisce l'intero ciclo di vita di un esperimento:
    1. Setup ottimizzatore e loss
    2. Loop delle epoche
    3. Early Stopping (salvataggio del modello migliore)
    4. Valutazione finale su Test set
    """
    # CrossEntropyLoss combina LogSoftmax e NLLLoss.
    # È la loss standard per classificazione multiclasse.
    criterion = nn.CrossEntropyLoss()
    
    # Adam (Adaptive Moment Estimation): Converge spesso più velocemente di SGD
    # gestendo learning rate specifici per ogni parametro.
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'epoch_times': []}
    
    # Variabili per Early Stopping
    best_val_acc = 0.0
    best_epoch = 0
    best_model_state = None
    epochs_no_improve = 0
    
    start_total = time.time()
    
    for epoch in range(config['max_epochs']):
        start_epoch = time.time()
        
        # Training e Validazione
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - start_epoch
        
        # Logging
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch_times'].append(epoch_time)
        
        if verbose:
            print(f"  Epoch {epoch+1:2d} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | Time: {epoch_time:.1f}s")
        
        # LOGICA EARLY STOPPING
        # Controlliamo se l'accuracy in validazione è migliorata di almeno 'min_delta'
        improvement = val_acc - best_val_acc
        
        if improvement > config['min_delta']:
            # Sì: Salviamo questo stato come "migliore" e resettiamo il contatore
            best_val_acc = val_acc
            best_epoch = epoch
            # .copy() è importante per non salvare il riferimento ma i dati
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
            if verbose:
                print(f"    ↑ Nuovo best! Val Acc: {val_acc:.2f}%")
        else:
            # No: Incrementiamo il contatore
            epochs_no_improve += 1
        
        # Se non migliora per 'patience' epoche consecutive, fermiamo tutto.
        # Questo previene l'overfitting sul training set.
        if epochs_no_improve >= config['patience']:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1} (no improvement for {config['patience']} epochs)")
            break
    
    total_time = time.time() - start_total
    
    # TEST FINALE
    # È cruciale caricare i pesi della "best_epoch" e non dell'ultima epoca,
    # perché l'ultima epoca potrebbe aver iniziato a fare overfitting.
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    _, test_acc = validate(model, test_loader, criterion, device)
    
    # Calcolo tempo medio (escludendo la prima epoca che include overhead di inizializzazione CUDA/Cache)
    if len(history['epoch_times']) > 1:
        avg_epoch_time = sum(history['epoch_times'][1:]) / len(history['epoch_times'][1:])
        first_epoch_time = history['epoch_times'][0]
    else:
        avg_epoch_time = history['epoch_times'][0]
        first_epoch_time = avg_epoch_time
    
    # Pacchetto risultati
    results = {
        'experiment_name': experiment_name,
        'parameters': sum(p.numel() for p in model.parameters()),
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'val_test_gap': best_val_acc - test_acc, # Indicatore di overfitting sul validation set
        'epochs_to_convergence': best_epoch + 1,
        'total_epochs': len(history['train_loss']),
        'total_time': total_time,
        'avg_epoch_time': avg_epoch_time,
        'first_epoch_time': first_epoch_time,
        'history': history
    }
    
    return results


# =============================================================================
# FUNZIONI DI VISUALIZZAZIONE
# =============================================================================

def plot_comparison(all_results, save_path='comparison_plots.png'):
    """Genera grafici a barre e scatter plot per confrontare le metriche chiave."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Separa FCNN e CNN
    fcnn_results = {k: v for k, v in all_results.items() if k.startswith('fcnn')}
    cnn_results = {k: v for k, v in all_results.items() if k.startswith('cnn')}
    
    # 1. Test Accuracy Comparison
    ax = axes[0, 0]
    names = list(all_results.keys())
    accs = [all_results[n]['test_acc'] for n in names]
    colors = ['blue' if n.startswith('fcnn') else 'green' for n in names]
    bars = ax.barh(names, accs, color=colors, alpha=0.7)
    ax.set_xlabel('Test Accuracy (%)')
    ax.set_title('Test Accuracy: FCNN (blu) vs CNN (verde)')
    ax.set_xlim(min(accs)-2, 100) # Zoom sulla zona interessante
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{acc:.2f}%', va='center', fontsize=8)
    
    # 2. Parameters vs Accuracy (Trade-off efficienza)
    ax = axes[0, 1]
    for name, res in all_results.items():
        color = 'blue' if name.startswith('fcnn') else 'green'
        marker = 'o' if name.startswith('fcnn') else 's'
        ax.scatter(res['parameters']/1000, res['test_acc'], 
                   c=color, marker=marker, s=100, alpha=0.7, label=name)
    ax.set_xlabel('Parametri (migliaia)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Efficienza: Parametri vs Accuracy')
    ax.grid(True, alpha=0.3)
    
    # 3. Epochs to Convergence
    ax = axes[1, 0]
    names = list(all_results.keys())
    epochs = [all_results[n]['epochs_to_convergence'] for n in names]
    colors = ['blue' if n.startswith('fcnn') else 'green' for n in names]
    ax.barh(names, epochs, color=colors, alpha=0.7)
    ax.set_xlabel('Epoche alla convergenza')
    ax.set_title('Velocità di Convergenza (Epoche)')
    
    # 4. Training Time
    ax = axes[1, 1]
    names = list(all_results.keys())
    times = [all_results[n]['avg_epoch_time'] for n in names]
    colors = ['blue' if n.startswith('fcnn') else 'green' for n in names]
    ax.barh(names, times, color=colors, alpha=0.7)
    ax.set_xlabel('Tempo medio per epoca (s)')
    ax.set_title('Costo Computazionale (Tempo)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Grafici comparativi salvati in '{save_path}'")
    plt.show()


def plot_learning_curves(all_results, save_path='learning_curves.png'):
    """Plotta le curve di Training vs Validation Accuracy per diagnosticare overfitting."""
    n_models = len(all_results)
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, (name, res) in enumerate(all_results.items()):
        ax = axes[idx]
        history = res['history']
        epochs = range(1, len(history['train_acc']) + 1)
        
        ax.plot(epochs, history['train_acc'], 'b-', label='Train', alpha=0.7)
        ax.plot(epochs, history['val_acc'], 'r-', label='Val', alpha=0.7)
        # Linea verticale che indica il punto di Early Stopping
        ax.axvline(res['epochs_to_convergence'], color='g', linestyle='--', alpha=0.5, label='Best Model')
        
        ax.set_xlabel('Epoca')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f"{name}\nTest: {res['test_acc']:.2f}%")
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Nascondi assi vuoti
    for idx in range(len(all_results), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Curve di apprendimento salvate in '{save_path}'")
    plt.show()


def print_results_table(all_results):
    """Stampa tabella testuale riassuntiva per log e debug."""
    print("\n" + "=" * 110)
    print("TABELLA RIASSUNTIVA DEI RISULTATI")
    print("=" * 110)
    print(f"{'Modello':<25} {'Params':>10} {'Val Acc':>10} {'Test Acc':>10} {'Gap':>8} {'Epoche':>8} {'Tempo/ep':>10}")
    print("-" * 110)
    
    # Prima FCNN
    print("\nFCNN:")
    for name, res in all_results.items():
        if name.startswith('fcnn'):
            gap = res.get('val_test_gap', res['best_val_acc'] - res['test_acc'])
            gap_str = f"{gap:+.2f}%" if gap != 0 else "0.00%"
            print(f"  {name:<23} {res['parameters']:>10,} {res['best_val_acc']:>9.2f}% "
                  f"{res['test_acc']:>9.2f}% {gap_str:>8} {res['epochs_to_convergence']:>8} "
                  f"{res['avg_epoch_time']:>9.1f}s")
    
    # Poi CNN
    print("\nCNN:")
    for name, res in all_results.items():
        if name.startswith('cnn'):
            gap = res.get('val_test_gap', res['best_val_acc'] - res['test_acc'])
            gap_str = f"{gap:+.2f}%" if gap != 0 else "0.00%"
            print(f"  {name:<23} {res['parameters']:>10,} {res['best_val_acc']:>9.2f}% "
                  f"{res['test_acc']:>9.2f}% {gap_str:>8} {res['epochs_to_convergence']:>8} "
                  f"{res['avg_epoch_time']:>9.1f}s")
    
    # Analisi gap Val/Test
    # Un gap elevato significa che il modello ha "memorizzato" il validation set
    # (Validation Overfitting), quindi il test set è una stima più onesta.
    print("\n" + "-" * 110)
    gaps = [res.get('val_test_gap', res['best_val_acc'] - res['test_acc']) for res in all_results.values()]
    avg_gap = sum(gaps) / len(gaps)
    max_gap = max(gaps)
    min_gap = min(gaps)
    print(f"\n📊 ANALISI GAP VAL/TEST (Controllo Affidabilità Validation):")
    print(f"   Gap medio: {avg_gap:+.3f}%  |  Gap max: {max_gap:+.3f}%  |  Gap min: {min_gap:+.3f}%")
    
    if abs(max_gap) < 0.5:
        print("   ✅ Gap accettabile! Early stopping funziona correttamente.")
    else:
        print("   ⚠️  Gap elevato! Potrebbe esserci ancora overfitting sul validation set.")
    
    print("=" * 110)


# =============================================================================
# MAIN - ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Importazione dinamica dei moduli modello (fcnn.py e cnn.py)
    # Questo permette di tenere le classi dei modelli in file separati
    # ma di usarle come se fossero definite qui.
    sys.path.append('.')
    modello_fcnn = import_module('fcnn')
    modello_cnn = import_module('cnn')
    FCNN = modello_fcnn.FCNN
    CNN = modello_cnn.CNN
    
    # Setup Device (GPU o CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Selezione degli esperimenti da eseguire
    if TEST_MODE:
        print("\n" + "!" * 70)
        print("⚠️  MODALITÀ TEST ATTIVA!")
        print("Esecuzione limitata alle configurazioni critiche per debug.")
        print("!" * 70)
        
        # Filtra solo esperimenti con test_priority=True
        fcnn_to_run = {k: v for k, v in FCNN_EXPERIMENTS.items() if v.get('test_priority', False)}
        cnn_to_run = {k: v for k, v in CNN_EXPERIMENTS.items() if v.get('test_priority', False)}
    else:
        # Esecuzione completa per la relazione
        fcnn_to_run = FCNN_EXPERIMENTS
        cnn_to_run = CNN_EXPERIMENTS
    
    # -----------------------------------------------------------------
    # PIPELINE DATI (ETL: Extract, Transform, Load)
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PREPARAZIONE DATI")
    print("=" * 70)
    
    # Trasformazioni:
    # 1. ToTensor: Converte immagine PIL (0-255) in Tensor Float (0.0-1.0)
    # 2. Normalize: Standardizza i dati (media 0.1307, std 0.3081 per MNIST).
    #    La standardizzazione accelera la convergenza dell'ottimizzatore.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download Dataset
    train_full = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Split Train/Validation
    # È fondamentale separare un Validation set dal Training set per il tuning
    # degli iperparametri e per l'Early Stopping.
    generator = torch.Generator().manual_seed(COMMON_CONFIG['seed'])
    train_dataset, val_dataset = random_split(train_full, [50000, 10000], generator=generator)
    
    # DataLoader
    # Gestisce il batching, lo shuffling e il caricamento parallelo.
    # pin_memory=True accelera il trasferimento RAM -> VRAM.
    train_loader = DataLoader(
        train_dataset, batch_size=COMMON_CONFIG['batch_size'], shuffle=True,
        num_workers=COMMON_CONFIG['num_workers'], pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=COMMON_CONFIG['batch_size'], shuffle=False,
        num_workers=COMMON_CONFIG['num_workers'], pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=COMMON_CONFIG['batch_size'], shuffle=False,
        num_workers=COMMON_CONFIG['num_workers'], pin_memory=True, persistent_workers=True
    )
    
    print(f"Dataset Split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # -----------------------------------------------------------------
    # ESECUZIONE LOOP ESPERIMENTI
    # -----------------------------------------------------------------
    all_results = {}
    
    # 1. Esperimenti FCNN
    if fcnn_to_run:
        print("\n" + "=" * 70)
        print(f"ESPERIMENTI FCNN ({len(fcnn_to_run)} configurazioni)")
        print("=" * 70)
        
        for exp_name, exp_config in fcnn_to_run.items():
            print(f"\n▶ {exp_name}: {exp_config['description']}")
            
            # Istanziazione Modello
            model = FCNN(
                input_size=784,
                hidden_sizes=exp_config['hidden_sizes'],
                num_classes=10,
                activation=exp_config['activation']
            ).to(device)
            
            print(f"  Parametri: {model.count_parameters():,}")
            
            # Training
            results = run_experiment(
                model, train_loader, val_loader, test_loader,
                device, COMMON_CONFIG, exp_name, verbose=True
            )
            
            # Salvataggio risultati parziali
            results['config'] = {k: v for k, v in exp_config.items() if k != 'test_priority'}
            results['model_type'] = 'FCNN'
            all_results[exp_name] = results
            
            # Feedback immediato
            gap = results['val_test_gap']
            print(f"  ✓ Test Accuracy: {results['test_acc']:.2f}%")
            print(f"  ✓ Gap Val/Test: {gap:+.2f}% {'✅' if abs(gap) < 0.5 else '⚠️'}")
    
    # 2. Esperimenti CNN
    if cnn_to_run:
        print("\n" + "=" * 70)
        print(f"ESPERIMENTI CNN ({len(cnn_to_run)} configurazioni)")
        print("=" * 70)
        
        for exp_name, exp_config in cnn_to_run.items():
            print(f"\n▶ {exp_name}: {exp_config['description']}")
            
            # Istanziazione Modello
            model = CNN(
                num_conv_blocks=exp_config['num_conv_blocks'],
                filters=exp_config['filters'],
                kernel_size=exp_config['kernel_size']
            ).to(device)
            
            print(f"  Parametri: {model.count_parameters():,}")
            
            # Training
            results = run_experiment(
                model, train_loader, val_loader, test_loader,
                device, COMMON_CONFIG, exp_name, verbose=True
            )
            
            # Salvataggio risultati parziali
            results['config'] = {k: v for k, v in exp_config.items() if k != 'test_priority'}
            results['model_type'] = 'CNN'
            all_results[exp_name] = results
            
            # Feedback immediato
            gap = results['val_test_gap']
            print(f"  ✓ Test Accuracy: {results['test_acc']:.2f}%")
            print(f"  ✓ Gap Val/Test: {gap:+.2f}% {'✅' if abs(gap) < 0.5 else '⚠️'}")

    # -----------------------------------------------------------------
    # SALVATAGGIO E REPORTING
    # -----------------------------------------------------------------
    print_results_table(all_results)
    
    # Salvataggio JSON (utile per includere dati grezzi nella relazione LaTeX/Word)
    results_summary = {
        name: {k: v for k, v in res.items() if k != 'history'}
        for name, res in all_results.items()
    }
    results_summary['common_config'] = COMMON_CONFIG
    results_summary['timestamp'] = datetime.now().isoformat()
    results_summary['test_mode'] = TEST_MODE
    
    output_filename = 'experiment_results_test.json' if TEST_MODE else 'experiment_results.json'
    with open(output_filename, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n✓ Risultati salvati in '{output_filename}'")
    
    # Generazione Grafici
    plot_suffix = '_test' if TEST_MODE else ''
    plot_comparison(all_results, f'comparison_plots{plot_suffix}.png')
    plot_learning_curves(all_results, f'learning_curves{plot_suffix}.png')
    
    print("\n" + "=" * 70)
    print("ESECUZIONE COMPLETATA CON SUCCESSO")
    print("=" * 70)