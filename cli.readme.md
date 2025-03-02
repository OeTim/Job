# Job-Scheduling-Optimierung: Ausführungsbefehle
## Grundlegende Befehle
### Simulation
        # Standard-Simulation
        python main.py --mode simulate

        # Simulation mit spezifischer Strategie
        python main.py --mode simulate --strategy FIFO
        python main.py --mode simulate --strategy LIFO
        python main.py --mode simulate --strategy SPT

### Training
        # Standard-Training (1000 Episoden)
        python main.py --mode train

        # Training mit angepasster Episodenanzahl
        python main.py --mode train --episodes 500
        python main.py --mode train --episodes 2000

        # Training mit eigenen Daten
        python main.py --mode train --data production_data.json

### Evaluierung
        # Evaluierung eines trainierten Modells
        python main.py --mode evaluate --model models/best_model_20250302_185201_jobs_50_makespan_2311.pt

        

        # Neuestes Modell evaluieren
        python main.py --mode evaluate --model $(ls -t models/best_model_* | head -1)

### Vergleichen
        # Vergleich der Baseline-Strategien
        python main.py --mode compare

        # Vergleich mit trainiertem Modell
        python main.py --mode compare --model models/best_model_YYYYMMDD_HHMMSS_makespan_XXX.pt

        # Vergleich mit dem neuesten Modell
        python main.py --mode compare --model $(ls -t models/best_model_* | head -1)





# Synthetische Daten generieren und speichern
python main.py --generate_data

# Daten aus einer Datei laden
python main.py --load_data production_data.json

# Abhängigkeitsgraph visualisieren
python main.py --visualize_graph

# Produktion mit einer bestimmten Strategie simulieren
python main.py --simulate --strategy FIFO
python main.py --simulate --strategy LIFO
python main.py --simulate --strategy SPT

# Training eines PPO-Agenten starten
python main.py --train --episodes 1000

# Training mit angepassten Hyperparametern
python main.py --train --episodes 500 --lr 0.0001 --entropy 0.02

# Training fortsetzen von einem gespeicherten Modell
python main.py --train --load_model models/ppo_model_20230615_120000.pt

# Einen trainierten Agenten evaluieren
python main.py --evaluate --model models/ppo_model.pt

# Verschiedene Strategien vergleichen
python main.py --compare_strategies

# Einen trainierten Agenten mit Baseline-Strategien vergleichen
python main.py --compare_strategies --model models/ppo_model.pt


# Daten generieren, trainieren und evaluieren in einem Durchlauf
python main.py --generate_data --train --episodes 500 --evaluate

# Daten laden, Abhängigkeitsgraph visualisieren und simulieren
python main.py --load_data production_data.json --visualize_graph --simulate