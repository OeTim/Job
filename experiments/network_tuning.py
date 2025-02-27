import os
import json
import subprocess
from datetime import datetime

# Beste Lernrate aus vorherigem Experiment
best_lr = 0.0003  # Ersetzen Sie dies durch Ihre beste Lernrate

# Zu testende Netzwerk-Konfigurationen
network_configs = [
    {"d_model": 64, "nhead": 2, "num_layers": 1, "dim_feedforward": 256},
    {"d_model": 128, "nhead": 4, "num_layers": 2, "dim_feedforward": 512},
    {"d_model": 256, "nhead": 8, "num_layers": 3, "dim_feedforward": 1024},
]

# Basis-Konfiguration
base_config = {
    "gamma": 0.99,
    "policy_clip": 0.2,
    "n_epochs": 10,
    "gae_lambda": 0.95,
    "batch_size": 64,
    "lr": best_lr,
    "entropy_coef": 0.01,
    "n_episodes": 300,
    "max_steps": 100,
    "update_interval": 20,
    "eval_interval": 50
}

# Verzeichnis für Experimente erstellen
exp_dir = os.path.join(os.getcwd(), 'experiments', 'network_tuning')
os.makedirs(exp_dir, exist_ok=True)

# Für jede Netzwerk-Konfiguration ein Experiment durchführen
for i, net_config in enumerate(network_configs):
    # Konfiguration aktualisieren
    config = base_config.copy()
    config.update(net_config)
    
    # Konfigurationsdatei speichern
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = os.path.join(exp_dir, f"config_net_{i}_{timestamp}.json")
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Starte Training mit Netzwerk-Konfiguration {i+1}...")
    
    # Training starten
    cmd = [
        "python", 
        "/Users/digitalsprung/Desktop/2025/2025_PYTHON/JOB_SCHEDULING/main.py",
        "--mode", "train",
        "--episodes", str(config["n_episodes"]),
        "--config", config_path
    ]
    
    subprocess.run(cmd)
    
    print(f"Training mit Netzwerk-Konfiguration {i+1} abgeschlossen.\n")

print("Alle Netzwerk-Experimente abgeschlossen.")