# Durchlauf 1

## Was die Grafiken zeigen

Die Grafiken zeigen die Trainingsperformance über 300 Episoden:

1. **Linkes Diagramm (Makespan)**: Zeigt den Makespan (gesamte Bearbeitungszeit aller Jobs) im Verlauf des Trainings.
2. **Rechtes Diagramm (Reward)**: Zeigt die Gesamtbelohnungen pro Episode.

## Bewertung der Ergebnisse

Die Ergebnisse sind nicht optimal, und hier sind die Gründe:

1. **Fehlende Konvergenz**: Bei beiden Grafiken ist keine klare Verbesserung über die Zeit erkennbar. Ein erfolgreiches Training würde eine kontinuierliche Abnahme des Makespans und eine Zunahme der Rewards zeigen.

2. **Hohe Varianz**: Die Werte schwanken stark zwischen aufeinanderfolgenden Episoden, was auf Instabilität im Lernprozess hindeutet.

## Mögliche Ursachen für die Probleme

1. **Explorations-/Exploitationsbalance**:
   - Die Epsilon-Decay-Rate (0.997) könnte zu langsam sein, wodurch der Agent zu lange zufällige Aktionen wählt.
   - Die Temperatur bei der Aktionsauswahl (1.0 + self.entropy_coef * 5) könnte angepasst werden.

2. **Netzwerkarchitektur**:
   - Die Transformer-Architektur ist komplex und könnte für dieses Problem überdesignt sein.
   - Der Code enthält eine SimplerPPONetwork-Klasse, die nicht genutzt wird, aber möglicherweise besser geeignet wäre.

3. **Hyperparameter**:
   - Der Lernrate (0.0003) oder der Batch-Größe (64) könnten angepasst werden.
   - Die Anzahl der Epochen (10) für das PPO-Update ist möglicherweise zu hoch oder zu niedrig.

4. **Belohnungsstruktur**: 
   - Die negativen Reward-Werte (-6.3 bis -6.44) deuten darauf hin, dass der Agent für jeden Zeitschritt bestraft wird. Die Belohnungsstruktur könnte überarbeitet werden.

## Verbesserungsvorschläge

1. **Hyperparameter-Tuning**:
   - Lernrate experimentell anpassen (z.B. 0.0001 bis 0.001)
   - Epsilon schneller reduzieren (z.B. 0.99 statt 0.997)
   - Weniger Epochen pro Update verwenden (3-5 statt 10)

2. **Netzwerkarchitektur**:
   - Die SimplerPPONetwork-Implementierung testen
   - Komplexität der Transformer reduzieren (weniger Attention-Heads, kleinere Dimensionen)

3. **Belohnungsstruktur**:
   - Belohnung für Makespan-Verbesserungen verstärken
   - Klarere Signale für gute Entscheidungen geben

4. **Trainingsansatz**:
   - Curriculum Learning erwägen (mit einfacheren Problemen beginnen)
   - Erfahrungswiederholung (Experience Replay) implementieren

Die Ergebnisse zeigen zwar kein klares Lernverhalten, aber das ist bei komplexen RL-Problemen wie Job-Scheduling nicht ungewöhnlich. Mit methodischen Anpassungen könnte die Performance deutlich verbessert werden.


