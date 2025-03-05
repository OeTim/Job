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
        python3 main.py --mode evaluate --model models/best_model_20250302_185201_jobs_50_makespan_2311.pt

        

        # Neuestes Modell evaluieren
        python main.py --mode evaluate --model $(ls -t models/best_model_* | head -1)

### Vergleichen
        # Vergleich der Baseline-Strategien
        python main.py --mode compare

        # Vergleich mit trainiertem Modell
        python main.py --mode compare --model models/best_model_YYYYMMDD_HHMMSS_makespan_XXX.pt

        # Vergleich mit dem neuesten Modell
        python main.py --mode compare --model $(ls -t models/best_model_* | head -1)



