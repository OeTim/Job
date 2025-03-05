# Configuration settings
config = {
    "n_jobs": 50,           
    "min_ops": 2,           
    "max_ops": 5,           
    "machines": ["M1", "M2", "M3", "M4"],
    "materials": ["Material_A", "Material_B", "Material_C"],
    "tools": ["Werkzeug", "Öl", "Kühlmittel", "Schablone"],
    # Neuer Eintrag: Startmaterial für jede Maschine
    "machine_initial": {
        "M1": "Material_A",
        "M2": "Material_A",
        "M3": "Material_A",
        "M4": "Material_A"
    },
    # Neuer Eintrag: Umrüstungstabelle pro Maschine
    "changeover_times": {
        "M1": {
            ("Material_A", "Material_A"): 0,
            ("Material_A", "Material_B"): 5,
            ("Material_A", "Material_C"): 7,
            ("Material_B", "Material_A"): 6,
            ("Material_B", "Material_B"): 0,
            ("Material_B", "Material_C"): 8,
            ("Material_C", "Material_A"): 9,
            ("Material_C", "Material_B"): 4,
            ("Material_C", "Material_C"): 0
        },
        # Für M2, M3, M4 nehmen wir hier gleiche Werte an – du kannst sie individuell anpassen:
        "M2": {
            ("Material_A", "Material_A"): 0,
            ("Material_A", "Material_B"): 5,
            ("Material_A", "Material_C"): 7,
            ("Material_B", "Material_A"): 6,
            ("Material_B", "Material_B"): 0,
            ("Material_B", "Material_C"): 8,
            ("Material_C", "Material_A"): 9,
            ("Material_C", "Material_B"): 4,
            ("Material_C", "Material_C"): 0
        },
        "M3": {
            ("Material_A", "Material_A"): 0,
            ("Material_A", "Material_B"): 5,
            ("Material_A", "Material_C"): 7,
            ("Material_B", "Material_A"): 6,
            ("Material_B", "Material_B"): 0,
            ("Material_B", "Material_C"): 8,
            ("Material_C", "Material_A"): 9,
            ("Material_C", "Material_B"): 4,
            ("Material_C", "Material_C"): 0
        },
        "M4": {
            ("Material_A", "Material_A"): 0,
            ("Material_A", "Material_B"): 5,
            ("Material_A", "Material_C"): 7,
            ("Material_B", "Material_A"): 6,
            ("Material_B", "Material_B"): 0,
            ("Material_B", "Material_C"): 8,
            ("Material_C", "Material_A"): 9,
            ("Material_C", "Material_B"): 4,
            ("Material_C", "Material_C"): 0
        }
    }
}

rl_config = {
    # PPO-Hyperparameter
    "gamma": 0.99,              # Unverändert
    "policy_clip": 0.1,         # Reduziert von 0.2
    "n_epochs": 8,              # Erhöht von 5
    "gae_lambda": 0.95,         # Unverändert
    "batch_size": 64,           # Reduziert von 128
    "lr": 0.0003,              # Reduziert von 0.0005
    "entropy_coef": 0.01,  
    
    # Transformer-Parameter
    "d_model": 256,             # Increased model capacity
    "nhead": 8,                 # More attention heads
    "num_layers": 3,            # More transformer layers
    "dim_feedforward": 1024,    # Larger feedforward network
    
    # Trainingsparameter
    "n_episodes": 5000,         # Erhöht von 2000
    "max_steps": 300,           # Erhöht von 200
    "update_interval": 20,  
    "eval_interval": 25,        # More frequent evaluation
    "save_interval": 50,        # More frequent saving
}