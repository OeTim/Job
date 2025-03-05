# Features des PPO-Agenten für Job-Scheduling

Der PPO-Agent für Job-Scheduling nutzt einen Graph-basierten Ansatz, bei dem Operationen als Knoten und Beziehungen zwischen ihnen als Kanten dargestellt werden. Folgende Features werden für das Training verwendet:

## Knoten-Features (Operationen)

Die Knoten-Features werden in der `_create_state_graph`-Methode erstellt und in der Variable `node_features` gespeichert:

1. **Maschinen-Index (normalisiert)**: 
   - Der Index der Maschine, auf der die Operation ausgeführt wird (z.B. M1 → 0, M2 → 1, ...)
   - Wird normalisiert durch Division durch die Anzahl der Maschinen: `machine_idx / len(self.machines)`

2. **Material-Typ (normalisiert)**:
   - Der Index des Materials, das von der Operation produziert wird
   - Wird aus dem Dictionary `material_types` abgerufen: `{"Material_A": 0, "Material_B": 1, "Material_C": 2}`
   - Wird normalisiert durch Division durch die Anzahl der Materialtypen: `material_idx / len(self.material_types)`

3. **Status-Flag**:
   - Binärer Wert: 0 = ausstehend, 1 = abgeschlossen
   - `status = 1 if op_data['status'] == 'completed' else 0`

4. **Priorität (normalisiert)**:
   - Die Priorität des Jobs, zu dem die Operation gehört
   - Wird auf einen Wert zwischen 0,1 und 1,0 normalisiert: `priority = op_data['priority'] / 10.0`

5. **Bearbeitungszeit (normalisiert)**:
   - Die benötigte Zeit für die Operation
   - Wird auf einen Wert zwischen 0 und 1 normalisiert: `time_normalized = op_data['time'] / 60.0`

6. **Benötigte Hilfsmittel (One-Hot-Encoding)**:
   - Ein One-Hot-Vektor, der angibt, welche Werkzeuge für die Operation benötigt werden
   - Die möglichen Werkzeuge sind: "Kühlmittel", "Öl", "Werkzeug", "Schablone"
   - Für jedes Werkzeug wird ein Wert von 0 oder 1 gesetzt: `tools_feature[self.tools[tool]] = 1`

Die Gesamtdimension der Knoten-Features beträgt somit: 5 + Anzahl der Werkzeuge (4) = 9.

## Kanten-Features (Beziehungen)

Drei Arten von Beziehungen (Kanten) werden zwischen den Operationen modelliert, jede mit einem One-Hot-Encoding:

1. **Vorgängerbeziehungen**: 
   - Kodiert als `[1, 0, 0]`
   - Gibt an, dass eine Operation vor einer anderen ausgeführt werden muss

2. **Maschinenbeziehungen**:
   - Kodiert als `[0, 1, 0]`
   - Verbindet Operationen, die auf derselben Maschine ausgeführt werden

3. **Job-Beziehungen**:
   - Kodiert als `[0, 0, 1]`
   - Verbindet Operationen, die zum selben Job gehören

## Zusätzliche Information: Masken für verfügbare Operationen

Neben den Feature-Vektoren wird auch eine Maske verwendet, um verfügbare Operationen zu identifizieren:

- `operation_mask`: Eine boolesche Maske, die angibt, welche Operationen aktuell verfügbar sind (d.h. alle Vorgänger abgeschlossen sind und sie für die Planung in Betracht gezogen werden können)

## Netzwerkarchitektur

Der Agent verwendet für die Feature-Verarbeitung ein Graph-Netzwerk mit:

- Graph Attention Layers (GAT) mit mehreren Aufmerksamkeitsköpfen
- Einbettungsschichten für die Knoten-Features
- Dropout für Regularisierung

Diese Architektur ermöglicht es dem Agenten, die strukturellen Beziehungen zwischen den Operationen zu berücksichtigen und eine effiziente Scheduling-Strategie zu erlernen.