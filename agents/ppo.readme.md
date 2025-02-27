# PPO Agent für Job Scheduling - Dokumentation
## Übersicht
Diese Implementierung verwendet Proximal Policy Optimization (PPO) mit Transformer-Architekturen für die Optimierung von Job-Scheduling-Problemen.

## Klassen und Funktionen
### TransformerEncoder
- Zweck : Verarbeitet sequentielle Daten mit Self-Attention-Mechanismen
- Beispiel : Kodiert Job-Features wie Bearbeitungszeiten und Maschinenabhängigkeiten in einen latenten Raum
### PPONetwork
- Zweck : Neuronales Netzwerk mit Actor-Critic-Architektur und Transformer-Encodern
- Beispiel : Verarbeitet Job- und Maschinenfeatures, um optimale Scheduling-Entscheidungen zu treffen
### PPOMemory
- Zweck : Speichert Erfahrungen (Zustände, Aktionen, Belohnungen) für Batch-Training
- Beispiel : Sammelt Daten über Scheduling-Entscheidungen und deren Auswirkungen auf den Makespan
### PPOAgent
- Zweck : Hauptklasse für das Reinforcement Learning mit PPO-Algorithmus
- Beispiel : Lernt, Jobs so zu planen, dass der Makespan minimiert wird Methoden:
- choose_action : Wählt den nächsten auszuführenden Job basierend auf aktuellem Zustand
- learn : Aktualisiert das Netzwerk basierend auf gesammelten Erfahrungen
- _prepare_tensors : Konvertiert gesammelte Daten in PyTorch-Tensoren
- _compute_advantages : Berechnet Vorteilswerte für stabileres Training
- save_models/load_models : Speichert/lädt trainierte Modelle
## Schlüsselkonzepte
- Transformer-Architektur : Erfasst komplexe Beziehungen zwischen Jobs und Maschinen
- Actor-Critic : Actor wählt Jobs aus, Critic bewertet die Qualität der Entscheidungen
- PPO-Algorithmus : Stabileres Training durch begrenztes Policy-Update
- GAE (Generalized Advantage Estimation) : Reduziert Varianz beim Lernen
## Anwendung
Der Agent lernt, Jobs so zu planen, dass der Makespan (Gesamtfertigstellungszeit) minimiert wird, während er Einschränkungen wie Maschinenkapazitäten und Jobabhängigkeiten berücksichtigt.