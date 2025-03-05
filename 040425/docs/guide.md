# Guide: Job-Scheduling mit Reinforcement Learning

## Überblick
Diese Implementierung verwendet Reinforcement Learning mit Proximal Policy Optimization (PPO) zur Lösung eines komplexen Job-Scheduling-Problems in der Fertigung. Ein Graph Neural Network modelliert die Abhängigkeiten zwischen Operationen und trifft Planungsentscheidungen.

## Komponenten

### 1. JobSchedulingEnvironment
- Simuliert eine Fertigungsumgebung mit Maschinen, Jobs und Operationen
- Verfolgt den Zustand von Maschinen, Material und Operationen
- Berechnet Umrüstzeiten, Startzeiten und Endzeiten der Operationen
- Unterstützt verschiedene Heuristiken (FIFO, LIFO, SPT, Priority, Random)
- Generiert Graph-basierte Zustandsdarstellungen für das RL-Modell
- Implementiert eine komplexe Reward-Funktion mit 7 Komponenten

### 2. GraphTransformer
- Neural Network zur Verarbeitung von Graphen
- Nutzt Graph Attention Network (GAT) mit mehreren Aufmerksamkeitsköpfen
- Verarbeitet Knoten- und Kantenattribute für Entscheidungsfindung

### 3. PPOAgent
- Implementiert den PPO-Algorithmus mit Actor-Critic-Architektur
- Actor: Wählt Operationen für die Planung aus
- Critic: Schätzt den Wert des aktuellen Zustands
- Verwendet Generalized Advantage Estimation (GAE)
- Optimiert mit clipped PPO-Verlustfunktion

### 4. Trainings- und Evaluierungsfunktionen
- `train_ppo_agent`: Trainiert den Agenten über mehrere Episoden
- `evaluate_method`: Vergleicht RL und traditionelle Heuristiken
- `generate_synthetic_data`: Erzeugt realistische Testdaten

## Graph-Repräsentation
- **Knoten**: Operationen mit Features für Maschine, Material, Status, Priorität, Zeit und Werkzeuge
- **Kanten**: Drei Beziehungstypen (Vorgänger, gleiche Maschine, gleicher Job)
- **Masken**: Identifizieren verfügbare Operationen

## Reward-Funktionen
1. Effiziente Maschinennutzung (+0.1 × Auslastung)
2. Abschluss von Operationen (+0.2 × normalisierte Priorität)
3. Frühzeitige Fertigstellung (+0.15)
4. Minimale Umrüstzeiten (+0.1)
5. Strafe für Leerlaufzeiten (-0.1 × normalisierte Wartezeit)
6. Bonus für abgeschlossene Jobs (+0.3)
7. Balancierung der Arbeitslast (+0.15 × Ausgewogenheit)
8. Makespan-Bonus bei Abschluss aller Jobs (2000 × base_makespan/makespan)

## Hauptablauf
1. Synthetische Daten generieren (50 Jobs mit 1-5 Operationen)
2. Traditionelle Heuristiken evaluieren
3. PPO-Agenten mit Graph Neural Network trainieren
4. Ergebnisse vergleichen und visualisieren
5. Bestes Modell speichern und evaluieren

## Visualisierungen
- Gantt-Charts für die Visualisierung des Schedules
- Graphdarstellung des Zustands mit farbcodierten Knoten und Kanten
- Trainingsfortschrittsgrafiken (Makespan und Rewards)
- Vergleichende Balkendiagramme der verschiedenen Methoden