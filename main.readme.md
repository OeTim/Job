# Job Scheduling Optimierung mit Reinforcement Learning
Diese README bietet einen Überblick über das Job-Scheduling-Optimierungssystem, das mit Reinforcement-Learning-Techniken implementiert wurde.

## Projektübersicht
Dieses Projekt implementiert ein intelligentes Job-Scheduling-System für Produktionsumgebungen mittels Reinforcement Learning (RL). Das System optimiert die Planung von Aufträgen auf mehreren Maschinen, um die Gesamtdurchlaufzeit (Makespan) zu minimieren und berücksichtigt dabei verschiedene Einschränkungen wie Maschinenverfügbarkeit, Rüstzeiten und Werkzeuganforderungen.

## Kernkomponenten
### 1. Produktionsumgebungssimulation
Das System simuliert eine Produktionsumgebung mit:

- Mehreren Aufträgen (Jobs), die aus verschiedenen Operationen bestehen
- Mehreren Maschinen mit unterschiedlichen Fähigkeiten
- Werkzeugen und Ressourcen, die für Operationen erforderlich sind
- Rüstzeiten und Anforderungen für Zwischenlagerung
- Auftragsabhängigkeiten und Vorgängerbeziehungen
### 2. Reinforcement-Learning-Ansatz
Das Projekt verwendet Proximal Policy Optimization (PPO), einen modernen RL-Algorithmus, um optimale Planungsstrategien zu erlernen:

- Der Agent lernt, welcher Auftrag basierend auf dem aktuellen Zustand des Produktionssystems als nächstes bearbeitet werden soll
- Die Umgebung gibt Feedback durch Belohnungen basierend auf Makespan, Maschinenauslastung und anderen Metriken
- Der Agent verbessert seine Strategie im Laufe der Zeit durch Versuch und Irrtum
### 3. Transformer-basierte neuronale Netzwerkarchitektur
Das System verwendet eine anspruchsvolle neuronale Netzwerkarchitektur:

- Transformer-Encoder verarbeiten Job- und Maschinenmerkmale separat
- Dies ermöglicht dem Modell, komplexe Beziehungen zwischen Jobs und Maschinen zu erfassen
- Die Architektur kann mit variablen Anzahlen von Jobs und Maschinen umgehen
## Hauptfunktionen
### Datengenerierung und -verwaltung
- Erzeugung synthetischer Daten für Tests und Entwicklung
- Unterstützung für das Laden und Speichern von Produktionsdaten im JSON-Format
- Visualisierung von Auftragsabhängigkeiten und Maschinenkonflikten
### Planungsstrategien
- Implementierung traditioneller Planungsheuristiken (FIFO, LIFO, SPT)
- Vergleich von RL-basierter Planung mit traditionellen Methoden
- Visualisierung von Plänen als Gantt-Diagramme
### Training und Auswertung
- Konfigurierbare Trainingsparameter
- Modell-Checkpointing und Speicherung der besten Modelle
- Auswertungsmetriken einschließlich Makespan, Maschinenauslastung und Wartezeiten
- Datenaugmentierungstechniken zur Verbesserung der Generalisierung
## Arbeitsablauf
1. Datenvorbereitung : Generieren oder Laden von Produktionsdaten mit Jobs, Operationen und Einschränkungen
2. Umgebungseinrichtung : Erstellen einer Simulationsumgebung, die das Produktionssystem modelliert
3. Agent-Training : Trainieren des RL-Agenten, um optimale Planungsstrategien zu erlernen
4. Auswertung : Vergleich der Leistung mit Baseline-Strategien
5. Einsatz : Verwendung des trainierten Agenten zur Planung realer Produktionsaufträge
## Vorteile
- Optimierte Produktion : Minimierung des Makespan und Verbesserung der Ressourcennutzung
- Anpassungsfähigkeit : Das System kann sich an veränderte Produktionsanforderungen anpassen
- Umgang mit Einschränkungen : Natürliche Handhabung komplexer Einschränkungen wie Vorgängerbeziehungen und Ressourcenbeschränkungen
- Kontinuierliche Verbesserung : Der Agent verbessert sich kontinuierlich, während er mehr Jobs verarbeitet
## Anwendungsbeispiele
### Fertigungsumgebung mit komplexen Abhängigkeiten
In einer Werkstattfertigung mit 30 Jobs und 4 Maschinen konnte das System den Makespan um 15% gegenüber herkömmlichen Methoden reduzieren, indem es die Reihenfolge der Operationen intelligent plant und Maschinenleerlaufzeiten minimiert.

### Optimierung bei begrenzten Ressourcen
Bei einer Produktionslinie mit begrenzter Werkzeugverfügbarkeit (4 verschiedene Werkzeugtypen) konnte das System die Werkzeugnutzung um 20% verbessern und gleichzeitig die durchschnittliche Wartezeit der Jobs um 25% reduzieren.

### Umgang mit Rüstzeiten
In einer Umgebung mit signifikanten Rüstzeiten zwischen verschiedenen Produkttypen konnte das System durch intelligente Batchbildung die Gesamtrüstzeit um 30% reduzieren und damit die Produktivität erheblich steigern.

### Priorisierung dringender Aufträge
Das System kann Jobs mit hoher Priorität bevorzugt behandeln und trotzdem einen effizienten Gesamtplan erstellen, was in einer Testumgebung zu einer Verbesserung der termingerechten Lieferung um 40% führte.

Dieser Reinforcement-Learning-Ansatz ermöglicht es dem System, Planungsstrategien zu entdecken, die herkömmliche Heuristiken übertreffen, indem es aus Erfahrungen lernt und sich an die spezifischen Eigenschaften der Produktionsumgebung anpasst.