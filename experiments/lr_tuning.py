import os
import json
import subprocess
from datetime import datetime

# Zu testende Lernraten
learning_rates = [0.0001, 0.0003, 0.0005, 0.001, 0.003]

# Basis-Konfiguration
base_config = {
    "gamma": 0.99,
    "policy_clip": 0.2,
    "n_epochs": 10,
    "gae_lambda": 0.95,
    "batch_size": 64,
    "entropy_coef": 0.01,
    "d_model": 128,
    "nhead": 4,
    "num_layers": 2,
    "dim_feedforward": 512,
    "n_episodes": 300,
    "max_steps": 100,
    "update_interval": 20,
    "eval_interval": 50
}

# Verzeichnis für Experimente erstellen
exp_dir = os.path.join(os.getcwd(), 'experiments', 'lr_tuning')
os.makedirs(exp_dir, exist_ok=True)

# Für jede Lernrate ein Experiment durchführen
for lr in learning_rates:
    # Konfiguration aktualisieren
    config = base_config.copy()
    config["lr"] = lr
    
    # Konfigurationsdatei speichern
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = os.path.join(exp_dir, f"config_lr_{lr}_{timestamp}.json")
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Starte Training mit Lernrate {lr}...")
    
    # Training starten
    cmd = [
        "python", 
        "/Users/digitalsprung/Desktop/2025/2025_PYTHON/JOB_SCHEDULING/main.py",
        "--mode", "train",
        "--episodes", str(config["n_episodes"]),
        "--config", config_path
    ]
    
    subprocess.run(cmd)
    
    print(f"Training mit Lernrate {lr} abgeschlossen.\n")

print("Alle Lernraten-Experimente abgeschlossen.")