# Reward-Berechnung im Job-Scheduling-System

## Übersicht

In Reinforcement Learning ist der Reward ein entscheidender Teil, der dem Agenten Feedback gibt, wie gut seine Aktionen sind. Im vorliegenden Job-Scheduling-System ist der Reward sorgfältig gestaltet, um verschiedene Aspekte einer guten Planung zu berücksichtigen.

## Die Reward-Berechnung im Detail

Der Reward wird in der `step`-Methode der `JobSchedulingEnvironment`-Klasse berechnet. Hier ist eine einfache Erklärung jeder Komponente:

### Basis-Reward

```python
reward = -0.05  # Reduzierte Basisstrafe pro Schritt
```

**Erklärung:** Der Agent erhält eine kleine negative Grundstrafe (-0.05) für jeden Schritt. Dies motiviert den Agenten, das Ziel mit möglichst wenigen Schritten zu erreichen.

### 1. Maschinennutzung

```python
active_machines = sum(1 for m_time in self.machine_times.values() if m_time > self.current_time)
machine_utilization = active_machines / len(self.machines) if self.machines else 0
reward += 0.1 * machine_utilization
```

**Erklärung:**
1. `active_machines` zählt, wie viele Maschinen aktuell in Betrieb sind
2. `machine_utilization` berechnet den Prozentsatz der genutzten Maschinen (0-1)
3. Der Reward wird um 10% der Maschinenauslastung erhöht
4. **Ziel:** Der Agent soll möglichst viele Maschinen parallel nutzen

### 2. Priorität der Operation

```python
reward += 0.2 * (op_data['priority'] / 10.0)
```

**Erklärung:**
1. Jede Operation hat eine Priorität von 1-10
2. Diese wird auf einen Wert zwischen 0.1-1.0 normalisiert
3. Der Reward wird um 20% dieses Wertes erhöht
4. **Ziel:** Der Agent soll Jobs mit hoher Priorität bevorzugen

### 3. Frühzeitige Fertigstellung

```python
avg_completion_time = sum(self.machine_times.values()) / len(self.machines) if self.machines else 0
if end_time < avg_completion_time:
    reward += 0.15
```

**Erklärung:**
1. `avg_completion_time` ist die durchschnittliche Fertigstellungszeit aller Maschinen
2. Wenn die aktuelle Operation früher als dieser Durchschnitt fertig wird, gibt es einen Bonus von 0.15
3. **Ziel:** Schnelle Operationen sollen bevorzugt werden

### 4. Minimale Umrüstzeiten

```python
if setup_time < self.machine_setup_times[machine]['materialWechsel']:
    reward += 0.1
```

**Erklärung:**
1. Wenn die Umrüstzeit geringer ist als ein voller Materialwechsel, gibt es einen Bonus von 0.1
2. **Ziel:** Der Agent soll versuchen, Operationen so zu planen, dass weniger umgerüstet werden muss

### 5. Strafe für Leerlaufzeiten

```python
idle_time = start_time - max(self.current_time, earliest_start)
if idle_time > 0:
    reward -= 0.1 * (idle_time / 50.0)
```

**Erklärung:**
1. `idle_time` ist die Zeit, die eine Maschine unnötig wartet
2. Bei Leerlaufzeit wird eine Strafe verhängt, proportional zur Wartezeit
3. Der Wert wird durch 50.0 geteilt, um große Strafen zu vermeiden
4. **Ziel:** Maschinen sollen nicht unnötig leerlaufen

### 6. Job-Fertigstellung

```python
job_operations = self.job_operations[job_name]
completed_ops_count = sum(1 for op_name in job_operations if self.operations[op_name]['status'] == 'completed')
if completed_ops_count == len(job_operations):
    reward += 0.3
```

**Erklärung:**
1. Prüft, ob alle Operationen eines Jobs abgeschlossen sind
2. Wenn ja, gibt es einen signifikanten Bonus von 0.3
3. **Ziel:** Vollständige Jobs sollen fertiggestellt werden

### 7. Ausbalancierte Maschinenlast

```python
machine_loads = list(self.machine_times.values())
load_std_dev = np.std(machine_loads) if machine_loads else 0
max_std_dev = 500
normalized_std_dev = min(load_std_dev / max_std_dev, 1.0)
reward += 0.15 * (1.0 - normalized_std_dev)
```

**Erklärung:**
1. Berechnet die Standardabweichung der Maschinenauslastung
2. Normalisiert diese auf einen Wert zwischen 0-1
3. Je gleichmäßiger die Last verteilt ist (niedrigere Standardabweichung), desto höher der Bonus
4. **Ziel:** Die Arbeitslast soll gleichmäßig auf alle Maschinen verteilt werden

### Zusätzliche Belohnung bei Fertigstellung aller Jobs

Wenn alle Jobs abgeschlossen sind, gibt es einen besonders großen Bonus:

```python
if done:
    makespan = max([time for machine, time in self.machine_times.items()])
    reward += 5000 * (base_makespan / makespan)
    
    expected_makespan = sum(op_data['time'] for op_name, op_data in self.operations.items()) / len(self.machines)
    if makespan < expected_makespan:
        reward += 500
```

**Erklärung:**
1. `makespan` ist die Gesamtdauer aller Jobs (wann die letzte Maschine fertig wird)
2. Ein kürzeres Makespan bringt exponentiell mehr Belohnung
3. Wenn die Planung sogar besser als ein einfacher Durchschnittswert ist, gibt es einen Extrabonus von 500
4. **Ziel:** Den Agenten motivieren, eine möglichst kurze Gesamtdauer zu erreichen

## Zusammenfassung der Belohnungskomponenten

| Komponente | Belohnung | Ziel |
|------------|-----------|------|
| Basis | -0.05 | Effizienz fördern |
| Maschinennutzung | 0.1 × Nutzung | Parallele Nutzung fördern |
| Priorität | 0.2 × (Priorität/10) | Wichtige Jobs zuerst |
| Frühzeitige Fertigstellung | +0.15 | Schnelle Operationen bevorzugen |
| Minimale Umrüstung | +0.1 | Umrüstzeiten minimieren |
| Leerlaufzeiten | -0.1 × (Leerlauf/50) | Leerlaufzeiten vermeiden |
| Job-Fertigstellung | +0.3 | Vollständige Jobs abschließen |
| Ausbalancierte Last | 0.15 × (1-Standardabw.) | Gleichmäßige Auslastung |
| Fertigstellung aller Jobs | 2000 × (Basis/Makespan) | Gesamtdauer minimieren |
| Übertreffung der Erwartung | +500 | Exzellente Planung belohnen |

Diese vielfältigen Belohnungskomponenten helfen dem Agenten, eine optimale Scheduling-Strategie zu erlernen, die verschiedene wichtige Kriterien berücksichtigt.