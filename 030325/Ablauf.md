# Job-Shop Scheduling mit Reinforcement Learning

# Zusammenfassung: jobs.py

Diese Python-Datei dient der Verwaltung von Fertigungsjobs und ihren Operationen in einem Produktionsumfeld. Sie bietet Funktionen zum Erstellen, Laden und Validieren von Job-Daten im JSON-Format.

## Hauptfunktionen

- **Laden und Erstellen von Jobs**: 
  - Bestehende Job-Daten aus JSON-Dateien laden
  - Neue Job-Strukturen mit zufälligen Eigenschaften erstellen
  - Validierung von Job-Abhängigkeiten

- **Produktionsoperationen**:
  - Jeder Job besteht aus mehreren Operationen
  - Operationen haben Attribute wie Bearbeitungszeit, Maschine, Materialien
  - Operationen können Vorgänger haben, die zuerst abgeschlossen sein müssen

- **Produktionsplanung**:
  - Berechnung von Umrüstzeiten zwischen verschiedenen Materialien
  - Berücksichtigung von Faktoren wie Hilfsmitteln und Zwischenlagerung
  - Job-Priorisierung

## Typische Verwendung

Die Datei wird hauptsächlich zur Verwaltung von Fertigungsjobs verwendet, wobei sie:

1. Eine bestehende jobs.json-Datei lädt oder eine neue erstellt
2. Die Validität der Job-Abhängigkeiten überprüft
3. Umrüstzeiten für Maschinenwechsel berechnet
4. Backup-Kopien der Job-Daten erstellt

Diese Funktionalität könnte in einem größeren Produktionsplanungssystem oder einer Scheduling-Anwendung verwendet werden, um Fertigungsprozesse zu optimieren.

## Disjunktiven Graph erstellen
Warum machen wir das? Der disjunktive Graph ist eine mathematische Darstellung des Job-Shop Problems, die es uns ermöglicht, die komplexen Beziehungen zwischen Operationen und Maschinen zu modellieren und zu analysieren.

Wofür machen wir das? Der Graph dient als Grundlage für:

- Die Visualisierung des Problems
- Die Berechnung des kritischen Pfads
- Die Identifikation von Maschinenkonflikten
- Die Eingabe für unseren Graph Transformer
Wie machen wir das? Wir erstellen einen gerichteten Graphen mit NetworkX:

- Knoten : Jede Operation wird als Knoten dargestellt
- Konjunktive Kanten : Verbindungen zwischen Operationen innerhalb eines Jobs (feste Reihenfolge)
- Disjunktive Kanten : Verbindungen zwischen Operationen, die dieselbe Maschine nutzen (Konflikte)


        G = nx.DiGraph()
        # Füge Operationen als Knoten hinzu
        G.add_node("Job_1_Op1", type="operation", machine="M1", time=5)
        # Füge konjunktive Kanten hinzu (Reihenfolge innerhalb eines Jobs)
        G.add_edge("Job_1_Op1", "Job_1_Op2", type="conjunctive", weight=5)
        # Füge disjunktive Kanten hinzu (Maschinenkonflikte)
        G.add_edge("Job_1_Op1", "Job_2_Op1", type="disjunctive", machine="M1")


## 3. Environment erstellen
Warum machen wir das? Ein Reinforcement Learning Environment ist notwendig, um den Agenten zu trainieren. Es definiert die Interaktion zwischen dem Agenten und dem Job-Shop Problem.

Wofür machen wir das? Das Environment ermöglicht:

- Die Simulation von Scheduling-Entscheidungen
- Die Bewertung von Entscheidungen durch Belohnungen
- Das Training des RL-Agenten unter realistischen Bedingungen
- Die Evaluation verschiedener Scheduling-Strategien
Wie machen wir das? Wir implementieren eine Gym-kompatible Umgebung mit:

- Zustandsraum : Repräsentation des aktuellen Schedules, der Maschinenauslastung und verfügbaren Operationen


        observation = {
            'remaining_ops': 5,
            'machine_utilization': [0.8, 0.6, 0.7],
            'critical_path_length': 25
        }
- Aktionsraum : Auswahl der Scheduling-Heuristik (FIFO, LIFO, SPT)
- Belohnungsfunktion : Basierend auf Makespan-Reduktion und Ressourceneffizienz
- Übergangsfunktion : Simulation der Auswirkungen von Scheduling-Entscheidungen


## 4. Graph Transformer implementieren
## Warum machen wir das? 
Graph Transformer können komplexe Beziehungen in Graphstrukturen erfassen und sind daher ideal für die Analyse von disjunktiven Graphen geeignet.
## Wofür machen wir das? Der Graph Transformer:
- Extrahiert relevante Merkmale aus dem disjunktiven Graphen
- Berücksichtigt sowohl lokale als auch globale Strukturen
- Unterstützt die Entscheidungsfindung des RL-Agenten
## Wie machen wir das? Wir implementieren einen Transformer mit:

- Knoteneinbettungen für Operationen
- Aufmerksamkeitsmechanismen für Graphstrukturen
- Positionscodierung für zeitliche Aspekte

## 5. PPO-Agent trainieren
## Warum machen wir das? 
Proximal Policy Optimization (PPO) ist ein stabiler RL-Algorithmus, der effizient für komplexe Entscheidungsprobleme trainiert werden kann.
## Wofür machen wir das? Der PPO-Agent:
- Lernt optimale Scheduling-Strategien
- Passt sich an verschiedene Probleminstanzen an
- Maximiert die Gesamtbelohnung (minimiert Makespan)
### Wie machen wir das? Wir implementieren:
- Eine Policy-Netzwerk mit dem Graph Transformer
- Ein Value-Netzwerk zur Zustandsbewertung
- Den PPO-Trainingsalgorithmus mit Clipping
- Evaluationsmetriken zur Fortschrittsmessung

## 6. Modell evaluieren und optimieren
### Warum machen wir das? 
Die Evaluation ist entscheidend, um die Leistung unseres Ansatzes zu messen und zu verbessern.
### Wofür machen wir das? Die Evaluation:

- Vergleicht unseren Ansatz mit klassischen Heuristiken
- Identifiziert Schwachstellen und Verbesserungspotenzial
- Validiert die Generalisierbarkeit auf neue Probleminstanzen
### Wie machen wir das? Wir führen durch:

- Tests auf Benchmark-Datensätzen
- Vergleiche mit etablierten Lösungsansätzen
- Analyse der gelernten Strategien
- Hyperparameter-Optimierung




