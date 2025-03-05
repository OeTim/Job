# Produktionsplanungsheuristiken - Einfach erklärt

In der Produktionsplanung (Scheduling) werden verschiedene Heuristiken verwendet, um zu entscheiden, welche Operation als nächstes ausgeführt werden soll. Diese Methoden helfen dabei, den Produktionsablauf zu optimieren und die Gesamtproduktionszeit (Makespan) zu minimieren.

## Die vier grundlegenden Heuristiken

### 1. FIFO (First In, First Out)

**Funktionsweise:** Operationen werden in der Reihenfolge ausgeführt, in der sie in die Warteschlange kommen.

**Beispiel:**  
Angenommen, folgende Operationen sind verfügbar:
- Operation A (wartet seit 10 Minuten)
- Operation B (wartet seit 5 Minuten)
- Operation C (wartet seit 2 Minuten)

Bei FIFO wird Operation A zuerst ausgeführt, dann B, dann C.

**Implementierung:**
```python
def _select_operation_heuristic(self):
    if self.heuristic == 'FIFO':
        # Wähle die erste Operation in der Queue
        return available_ops[0]['name']
```

**Vorteile:** Einfach, fair (keine Operation wird endlos zurückgestellt)  
**Nachteile:** Berücksichtigt keine Bearbeitungszeiten oder Prioritäten

### 2. LIFO (Last In, First Out)

**Funktionsweise:** Die zuletzt verfügbar gewordenen Operationen werden zuerst ausgeführt.

**Beispiel:**  
Mit den gleichen verfügbaren Operationen:
- Operation A (wartet seit 10 Minuten)
- Operation B (wartet seit 5 Minuten)
- Operation C (wartet seit 2 Minuten)

Bei LIFO wird Operation C zuerst ausgeführt, dann B, dann A.

**Implementierung:**
```python
def _select_operation_heuristic(self):
    if self.heuristic == 'LIFO':
        # Wähle die letzte Operation in der Queue
        return available_ops[-1]['name']
```

**Vorteile:** Kann in bestimmten Situationen zu weniger Umrüstvorgängen führen  
**Nachteile:** Kann zu Verhungern (starvation) von älteren Operationen führen

### 3. SPT (Shortest Processing Time)

**Funktionsweise:** Operationen mit der kürzesten Bearbeitungszeit werden zuerst ausgeführt.

**Beispiel:**  
Angenommen, folgende Operationen sind verfügbar:
- Operation A (Bearbeitungszeit: 30 Minuten)
- Operation B (Bearbeitungszeit: 15 Minuten)
- Operation C (Bearbeitungszeit: 45 Minuten)

Bei SPT wird Operation B zuerst ausgeführt, dann A, dann C.

**Implementierung:**
```python
def _select_operation_heuristic(self):
    if self.heuristic == 'SPT':
        # Wähle die Operation mit der kürzesten Bearbeitungszeit
        return min(available_ops, key=lambda op: op['data']['time'])['name']
```

**Vorteile:** Minimiert die durchschnittliche Durchlaufzeit; viele kurze Jobs können schnell abgeschlossen werden  
**Nachteile:** Lange Operationen können zurückgestellt werden

### 4. Random

**Funktionsweise:** Eine zufällige Operation wird aus den verfügbaren Operationen ausgewählt.

**Beispiel:**  
Mit den gleichen verfügbaren Operationen A, B und C hat jede Operation die gleiche Wahrscheinlichkeit, ausgewählt zu werden.

**Implementierung:**
```python
def _select_operation_heuristic(self):
    # Zufällige Auswahl
    return random.choice(available_ops)['name']
```

**Vorteile:** Kann als Baseline für Vergleiche dienen; verhindert systematische Probleme, die bei deterministischen Methoden auftreten können  
**Nachteile:** Bietet keine gezielte Optimierung

## Wie diese Heuristiken in der Praxis angewendet werden

Im Produktionsplanungssystem werden diese Heuristiken wie folgt angewendet:

1. Das System identifiziert zunächst alle Operationen, die **derzeit verfügbar** sind (deren Vorgängeroperationen bereits abgeschlossen sind).

2. Basierend auf der gewählten Heuristik wird eine Operation aus diesen verfügbaren Operationen ausgewählt.

3. Die ausgewählte Operation wird dann der entsprechenden Maschine zugewiesen und ausgeführt.

4. Nach Abschluss der Operation wird die Liste der verfügbaren Operationen aktualisiert, und der Prozess wiederholt sich.

## Ergebnisse der Heuristiken im Beispielprojekt

Die Ergebnisse aus der Auswertung zeigen folgende durchschnittliche Makespan-Werte:

- **FIFO**: 5841.00
- **LIFO**: 5563.00
- **SPT**: 5620.00
- **Random**: 5511.60

Im gegebenen Beispiel hat die **Random**-Heuristik überraschenderweise die besten Ergebnisse erzielt, gefolgt von **LIFO**. Dies kann darauf hindeuten, dass in diesem spezifischen Produktionsszenario komplexe Abhängigkeiten vorliegen, bei denen deterministischere Ansätze nicht optimal sind.

## Vergleich mit dem PPO-Agenten

Als Alternative zu den einfachen Heuristiken verwendet das System auch einen PPO-Agenten (Proximal Policy Optimization), der durch maschinelles Lernen eine optimale Planungsstrategie entwickelt. Dieser kann in komplexen Szenarien oft bessere Ergebnisse liefern als die klassischen Heuristiken.

Der PPO-Agent lernt durch Versuch und Irrtum, welche Operationen in welcher Reihenfolge ausgeführt werden sollten, und berücksichtigt dabei mehr Faktoren als die einfachen Heuristiken, wie z.B.:

- Maschinenauslastung
- Materialwechsel auf Maschinen
- Gesamtproduktionszeit
- Prioritäten der Jobs
- Abhängigkeiten zwischen Operationen

Dies ermöglicht eine intelligentere und anpassungsfähigere Planungsstrategie.