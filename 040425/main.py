import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
import torch.optim as optim
from tqdm import tqdm

# Seed setzen für Reproduzierbarkeit
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Gerätekonfiguration (GPU falls verfügbar, sonst CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class JobSchedulingEnvironment:
    """
    Umgebung für das Job-Scheduling-Problem
    """
    def __init__(self, data_file, heuristic=None):
        # Daten laden
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Jobs und Maschinen extrahieren
        self.jobs = self.data['jobs']
        self.machine_setup_times = self.data['maschinenUmrüstzeiten']
        
        # Liste aller Maschinen erstellen
        self.machines = list(self.machine_setup_times.keys())
        
        # Mapping von Materialtypen zu IDs
        self.material_types = {"Material_A": 0, "Material_B": 1, "Material_C": 2}
        
        # Mapping von Hilfsmitteln zu IDs
        self.tools = {"Kühlmittel": 0, "Öl": 1, "Werkzeug": 2, "Schablone": 3}
        
        # Heuristik für Entscheidungen, falls kein RL-Agent verwendet wird
        self.heuristic = heuristic  # 'FIFO', 'LIFO', 'SPT', 'Random'
        
        # Initialisiere Umgebung
        self.reset()
    
    def reset(self):
        """
        Setzt die Umgebung zurück und gibt den initialen Zustand zurück
        """
        # Aktuelle Zeit für jede Maschine (wann sie wieder verfügbar ist)
        self.machine_times = {machine: 0 for machine in self.machines}
        
        # Aktuelles Material auf jeder Maschine
        self.machine_materials = {machine: None for machine in self.machines}
        
        # Queue mit verfügbaren Operationen (die noch keine Vorgänger haben oder deren Vorgänger abgeschlossen sind)
        self.operation_queue = []
        
        # Dictionary zum Speichern der Fertigstellungszeiten von Operationen
        self.completed_operations = {}
        
        # Abgeschlossene Jobs
        self.completed_jobs = set()
        
        # Operationen vorbereiten und initiale Operation Queue aufbauen
        self._prepare_operations()
        self._update_operation_queue()
        
        # Aktuelle Gesamtzeit
        self.current_time = 0
        
        # Aktionshistorie für Analysen
        self.action_history = []
        
        # Schedule für visuelle Darstellung
        self.schedule = {machine: [] for machine in self.machines}
        
        # Graph für den aktuellen Zustand erstellen
        self.state = self._create_state_graph()
        
        return self.state
    
    def _prepare_operations(self):
        """
        Bereitet die Operationen für das Scheduling vor
        """
        # Für jede Operation:
        # - Eindeutige ID zuweisen
        # - Status (verfügbar, laufend, abgeschlossen)
        # - Abhängigkeiten zu anderen Operationen verfolgen
        
        self.operations = {}
        self.job_operations = defaultdict(list)
        self.operation_dependencies = defaultdict(list)
        self.dependent_operations = defaultdict(list)
        
        # Zuerst alle Operationen sammeln und mit IDs versehen
        for job in self.jobs:
            job_name = job['Name']
            
            for op in job['Operationen']:
                op_name = op['Name']
                self.operations[op_name] = {
                    'job': job_name,
                    'time': op['benötigteZeit'],
                    'machine': op['Maschine'],
                    'predecessors': op.get('Vorgänger', []),
                    'material': op['produziertesMaterial'],
                    'tools': op.get('benötigteHilfsmittel', []),
                    'setup_cost': op.get('umruestkosten', 0),
                    'buffer': op.get('zwischenlager', None),
                    'status': 'pending',
                    'priority': job['Priorität']
                }
                
                self.job_operations[job_name].append(op_name)
        
        # Dann Abhängigkeiten einrichten
        for op_name, op_data in self.operations.items():
            for pred in op_data['predecessors'] or []:
                if pred:
                    self.operation_dependencies[op_name].append(pred)
                    self.dependent_operations[pred].append(op_name)
    
    def _update_operation_queue(self):
        """
        Aktualisiert die Queue der verfügbaren Operationen
        """
        self.operation_queue = []
        
        for op_name, op_data in self.operations.items():
            if op_data['status'] == 'pending':
                # Überprüfen, ob alle Vorgänger abgeschlossen sind
                all_predecessors_completed = True
                earliest_start_time = 0
                
                for pred in op_data['predecessors'] or []:
                    if pred not in self.completed_operations:
                        all_predecessors_completed = False
                        break
                    pred_completion_time = self.completed_operations[pred]
                    
                    # Wenn es eine Mindestverweildauer im Zwischenlager gibt
                    pred_op_data = self.operations[pred]
                    if pred_op_data.get('buffer'):
                        min_dwell_time = pred_op_data['buffer'].get('minVerweildauer', 0)
                        pred_completion_time += min_dwell_time
                    
                    earliest_start_time = max(earliest_start_time, pred_completion_time)
                
                # Wenn alle Vorgänger abgeschlossen sind, füge die Operation zur Queue hinzu
                if all_predecessors_completed:
                    self.operation_queue.append({
                        'name': op_name,
                        'data': op_data,
                        'earliest_start': earliest_start_time
                    })
    
    def _select_operation_heuristic(self):
        """
        Wählt eine Operation basierend auf einer Heuristik aus
        """
        if not self.operation_queue:
            return None
        
        # Filtere Operationen, die jetzt gestartet werden können
        available_ops = [op for op in self.operation_queue 
                         if op['earliest_start'] <= self.current_time]
        
        if not available_ops:
            # Wenn keine Operation jetzt verfügbar ist, springe zur nächstmöglichen Zeit
            next_available_time = min([op['earliest_start'] for op in self.operation_queue])
            self.current_time = next_available_time
            available_ops = [op for op in self.operation_queue 
                             if op['earliest_start'] <= self.current_time]
        
        if self.heuristic == 'FIFO':
            # First In, First Out: Wähle die erste Operation in der Queue
            return available_ops[0]['name']
        
        elif self.heuristic == 'LIFO':
            # Last In, First Out: Wähle die letzte Operation in der Queue
            return available_ops[-1]['name']
        
        elif self.heuristic == 'SPT':
            # Shortest Processing Time: Wähle die Operation mit der kürzesten Bearbeitungszeit
            return min(available_ops, key=lambda op: op['data']['time'])['name']
        
        elif self.heuristic == 'Priority':
            # Highest Priority: Wähle die Operation mit der höchsten Priorität
            return max(available_ops, key=lambda op: op['data']['priority'])['name']
        
        else:  # 'Random' oder undefiniert
            # Zufällige Auswahl
            return random.choice(available_ops)['name']
    
    def step(self, action=None):
        """
        Führt einen Schritt in der Umgebung aus
        
        Args:
            action: Index der ausgewählten Operation oder None für Heuristik-basierte Auswahl
            
        Returns:
            next_state: Der nächste Zustand
            reward: Die Belohnung für die Aktion
            done: True, wenn alle Jobs abgeschlossen sind
            info: Zusätzliche Informationen
        """
        if not self.operation_queue:
            # Keine Operationen mehr verfügbar
            return self.state, 0, True, {"makespan": self.current_time}
        
        # Operation auswählen (entweder über Aktion oder Heuristik)
        if action is not None:
            # Aktion ist ein Index in der operation_queue
            if 0 <= action < len(self.operation_queue):
                selected_op_name = self.operation_queue[action]['name']
            else:
                # Bei ungültigem Index default zur ersten Operation
                selected_op_name = self.operation_queue[0]['name']
        else:
            # Heuristik verwenden
            selected_op_name = self._select_operation_heuristic()
        
        if selected_op_name is None:
            # Keine Operation konnte ausgewählt werden
            return self.state, 0, False, {}
        
        # Ausgewählte Operation finden
        selected_op = None
        for op in self.operation_queue:
            if op['name'] == selected_op_name:
                selected_op = op
                break
        
        if not selected_op:
            return self.state, 0, False, {}
        
        # Operation aus der Queue entfernen
        self.operation_queue = [op for op in self.operation_queue if op['name'] != selected_op_name]
        
        # Daten der Operation extrahieren
        op_data = selected_op['data']
        machine = op_data['machine']
        op_time = op_data['time']
        op_material = op_data['material']
        job_name = op_data['job']
        earliest_start = selected_op['earliest_start']
        
        # Startzeit bestimmen (Maximum aus aktuellem Zeitpunkt, Maschinenbereitschaft und frühestem Start)
        start_time = max(self.current_time, self.machine_times[machine], earliest_start)
        
        # Umrüstzeit berechnen
        setup_time = 0
        if self.machine_materials[machine] != op_material:
            # Materialwechsel
            setup_time = self.machine_setup_times[machine]['materialWechsel']
        else:
            # Standardumrüstung
            setup_time = self.machine_setup_times[machine]['standardZeit']
        
        # Startzeit um Umrüstzeit erhöhen
        start_time += setup_time
        
        # Endzeit berechnen
        end_time = start_time + op_time
        
        # Maschine aktualisieren
        self.machine_times[machine] = end_time
        self.machine_materials[machine] = op_material
        
        # Operation als abgeschlossen markieren
        self.operations[selected_op_name]['status'] = 'completed'
        self.completed_operations[selected_op_name] = end_time
        
        # Schedule aktualisieren
        self.schedule[machine].append({
            'operation': selected_op_name,
            'job': job_name,
            'start': start_time,
            'end': end_time,
            'setup': setup_time
        })
        
        # Aktion zur Historie hinzufügen
        self.action_history.append({
            'time': self.current_time,
            'operation': selected_op_name,
            'machine': machine,
            'start': start_time,
            'end': end_time
        })
        
        # Aktuelle Zeit aktualisieren (falls nötig)
        self.current_time = max(self.current_time, start_time)
        
        # Operation Queue aktualisieren
        self._update_operation_queue()
        
        # Überprüfen, ob alle Operationen eines Jobs abgeschlossen sind
        for job in self.jobs:
            job_name = job['Name']
            if job_name not in self.completed_jobs:
                all_ops_completed = all(
                    self.operations[op_name]['status'] == 'completed' 
                    for op_name in self.job_operations[job_name]
                )
                if all_ops_completed:
                    self.completed_jobs.add(job_name)
                # Vereinfachte Reward-Funktion mit Fokus auf Makespan-Minimierung
        reward = -0.01  # Kleinere Basisstrafe pro Schritt, um lange Lösungswege zu vermeiden
        
        # 1. Belohne effiziente Maschinennutzung
        # Berechnung: Zähle Maschinen, deren Endzeit (machine_times) größer als die aktuelle Zeit ist
        # Diese Maschinen sind aktuell mit Operationen belegt und daher "aktiv"
        # Teile durch Gesamtzahl der Maschinen, um einen Wert zwischen 0 und 1 zu erhalten
        # Höherer Wert = mehr Maschinen parallel aktiv = bessere Ressourcennutzung
        active_machines = sum(1 for m_time in self.machine_times.values() if m_time > self.current_time)
        machine_utilization = active_machines / len(self.machines) if self.machines else 0
        reward += 0.05 * machine_utilization  # Skalierungsfaktor 0.05 bestimmt Einfluss dieses Teilrewards
        
        # 2. Belohne Operationen mit höherer Priorität
        # Berechnung: Nimm die Priorität des Jobs (1-10) und normalisiere auf 0.1-1.0
        # Höhere Priorität = höherer Reward, um wichtigere Jobs zu bevorzugen
        # Die Priorität kommt direkt aus den Jobdaten und wurde bei _prepare_operations() übernommen
        reward += 0.1 * (op_data['priority'] / 10.0)  # Skalierungsfaktor 0.1 bestimmt Einfluss
        
        # 3. Strafe für lange Wartezeiten bei verfügbaren Maschinen
        # Berechnung: Differenz zwischen tatsächlicher Startzeit und frühestmöglicher Startzeit
        # start_time = Zeitpunkt, zu dem die Operation tatsächlich beginnt (inkl. Umrüstzeit)
        # earliest_start = Frühester möglicher Start (basierend auf Vorgängeroperationen)
        # current_time = Aktuelle Simulationszeit
        # Wenn die Operation später als nötig startet, gibt es eine Strafe proportional zur Wartezeit
        idle_time = start_time - max(self.current_time, earliest_start)
        if idle_time > 0:
            reward -= 0.05 * (idle_time / 50.0)  # Division durch 50.0 normalisiert die Wartezeit
        
        # 4. Belohnung für wenige Umrüstungen (neu hinzugefügt)
        # Berechnung: Vergleiche das aktuelle Material auf der Maschine mit dem benötigten Material
        # Wenn kein Materialwechsel nötig ist, gibt es einen Bonus, da Umrüstzeiten eingespart werden
        # self.machine_materials[machine] = Aktuelles Material auf der Maschine
        # op_material = Von der Operation benötigtes Material
        if self.machine_materials[machine] == op_material:
            # Keine Materialumrüstung notwendig - gib Bonus
            reward += 0.08  # Direkter Bonus für vermiedene Umrüstung
        else:
            # Materialwechsel notwendig - kleine Strafe
            reward -= 0.03  # Kleine Strafe für notwendige Umrüstung
        
        # Fertig, wenn alle Jobs abgeschlossen sind
        done = len(self.completed_jobs) == len(self.jobs)
        
        if done:
            # Makespan (Gesamtfertigstellungszeit) berechnen
            # Berechnung: Maximum aller Maschinenendzeiten = Zeitpunkt, zu dem alle Operationen fertig sind
            makespan = max([time for machine, time in self.machine_times.items()])
            
            # Abschlussbelohnung: Fester Bonus für das erfolgreiche Abschließen aller Jobs
            # Dies hilft dem Agenten, vollständige Schedules zu bevorzugen
            reward += 5.0
            
            # Hier könnte später ein Vergleich mit Heuristiken eingebaut werden
        
        # Neuen Zustand erstellen
        self.state = self._create_state_graph()
        
        return self.state, reward, done, {"makespan": max(self.machine_times.values()) if done else 0}
    def _create_state_graph(self):
        """
        Erstellt einen Graph für den aktuellen Zustand
        
        Returns:
            state_graph: Ein Graph, der den aktuellen Zustand repräsentiert
        """
        # Knoten und Kanten für den Graphen erstellen
        nodes = []
        edges = []
        node_features = []
        edge_features = []
        
        # Mapping von Operationsnamen zu Knotenindizes
        op_to_idx = {}
        
        # Zunächst alle Operationen als Knoten hinzufügen
        idx = 0
        for op_name, op_data in self.operations.items():
            op_to_idx[op_name] = idx
            
            # Knoteneigenschaften
            machine_idx = int(op_data['machine'][1:]) - 1  # M1 -> 0, M2 -> 1, ...
            material_idx = self.material_types[op_data['material']]
            
            # Status: 0 = ausstehend, 1 = abgeschlossen
            status = 1 if op_data['status'] == 'completed' else 0
            
            # Feature für Werkzeuge (one-hot encoding)
            tools_feature = [0] * len(self.tools)
            for tool in op_data.get('tools', []):
                tools_feature[self.tools[tool]] = 1
            
            # Priorität normalisieren (1-10 -> 0.1-1.0)
            priority = op_data['priority'] / 10.0
            
            # Zeit normalisieren (0-60 -> 0-1)
            time_normalized = op_data['time'] / 60.0
            
            # Alle Features zusammenfügen
            features = [
                machine_idx / len(self.machines),  # Maschine
                material_idx / len(self.material_types),  # Material
                status,  # Status
                priority,  # Priorität
                time_normalized,  # Zeit
            ] + tools_feature  # Werkzeuge
            
            node_features.append(features)
            nodes.append(idx)
            idx += 1
        
        # Dann Kanten für Vorgängerbeziehungen hinzufügen
        for op_name, op_data in self.operations.items():
            src_idx = op_to_idx[op_name]
            
            # Vorgängerbeziehungen
            for pred in op_data.get('predecessors', []) or []:
                if pred:
                    dest_idx = op_to_idx[pred]
                    edges.append([dest_idx, src_idx])  # Vorgänger -> Operation
                    edge_features.append([1, 0, 0])  # Kantentyp: Vorgänger
            
            # Maschinenbeziehungen (Operationen auf der gleichen Maschine)
            for other_op, other_data in self.operations.items():
                if other_op != op_name and other_data['machine'] == op_data['machine']:
                    other_idx = op_to_idx[other_op]
                    edges.append([src_idx, other_idx])  # Operation -> andere Operation auf gleicher Maschine
                    edge_features.append([0, 1, 0])  # Kantentyp: Gleiche Maschine
            
            # Jobbeziehungen (Operationen im gleichen Job)
            for other_op, other_data in self.operations.items():
                if other_op != op_name and other_data['job'] == op_data['job']:
                    other_idx = op_to_idx[other_op]
                    edges.append([src_idx, other_idx])  # Operation -> andere Operation im gleichen Job
                    edge_features.append([0, 0, 1])  # Kantentyp: Gleicher Job
        
        # Graph-Daten zusammenstellen
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        # Zusätzliche Informationen
        operation_mask = torch.zeros(len(nodes), dtype=torch.bool)
        for i, op in enumerate(self.operation_queue):
            op_idx = op_to_idx[op['name']]
            operation_mask[op_idx] = True
        
        # Data-Objekt erstellen
        graph_data = {
            'x': x,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'operation_mask': operation_mask,
            'operation_queue': self.operation_queue,
            'op_to_idx': op_to_idx
        }
        
        return graph_data
    
    def render(self, mode='human'):
        """
        Visualisiert den aktuellen Zustand
        """
        if mode == 'human':
            # Gantt-Chart für das aktuelle Schedule erstellen
            self._create_gantt_chart()
        elif mode == 'graph':
            # Graph visualisieren
            self._visualize_graph()
    
    def _create_gantt_chart(self):
        """
        Erstellt ein Gantt-Chart für das Schedule
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Farben für Jobs
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.jobs)))
        job_color = {job['Name']: colors[i] for i, job in enumerate(self.jobs)}
        
        # Y-Positionen für Maschinen
        y_positions = {machine: i for i, machine in enumerate(self.machines)}
        
        # Zeichne Balken für jede Operation
        for machine, operations in self.schedule.items():
            for op in operations:
                job = op['job']
                start = op['start']
                duration = op['end'] - op['start']
                setup_time = op['setup']
                
                # Operation zeichnen
                rect = ax.barh(
                    y_positions[machine],
                    duration,
                    left=start,
                    height=0.5,
                    color=job_color[job],
                    alpha=0.8
                )
                
                # Setup-Zeit markieren
                if setup_time > 0:
                    ax.barh(
                        y_positions[machine],
                        setup_time,
                        left=start,
                        height=0.5,
                        color='red',
                        alpha=0.3
                    )
                
                # Labels hinzufügen
                ax.text(
                    start + duration/2,
                    y_positions[machine],
                    op['operation'],
                    ha='center',
                    va='center',
                    color='white'
                )
        
        # Achsen und Titel konfigurieren
        ax.set_yticks(list(y_positions.values()))
        ax.set_yticklabels(list(y_positions.keys()))
        ax.set_xlabel('Zeit')
        ax.set_ylabel('Maschine')
        ax.set_title('Job Scheduling Gantt-Chart')
        
        # Makespan anzeigen
        makespan = max([time for machine, time in self.machine_times.items()])
        ax.axvline(x=makespan, color='black', linestyle='--', alpha=0.7)
        ax.text(makespan, len(self.machines) - 0.5, f'Makespan: {makespan}', ha='right')
        
        # Legend für Jobs
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=job)
                           for job, color in job_color.items()]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.show()
    
    def _visualize_graph(self):
        """
        Visualisiert den Zustandsgraphen
        """
        G = nx.DiGraph()
        
        # Knoten hinzufügen
        op_names = list(self.operations.keys())
        for i, op_name in enumerate(op_names):
            op_data = self.operations[op_name]
            status = "completed" if op_data['status'] == 'completed' else "pending"
            
            # Markiere verfügbare Operationen
            is_available = False
            for op in self.operation_queue:
                if op['name'] == op_name:
                    is_available = True
                    break
            
            if is_available:
                status = "available"
            
            G.add_node(
                op_name,
                machine=op_data['machine'],
                job=op_data['job'],
                status=status,
                time=op_data['time']
            )
        
        # Kanten für Vorgängerbeziehungen hinzufügen
        for op_name, op_data in self.operations.items():
            for pred in op_data.get('predecessors', []) or []:
                if pred:
                    G.add_edge(pred, op_name, type='predecessor')
        
        # Kanten für Maschinenbeziehungen hinzufügen
        for op_name, op_data in self.operations.items():
            for other_op, other_data in self.operations.items():
                if other_op != op_name and other_data['machine'] == op_data['machine']:
                    G.add_edge(op_name, other_op, type='machine')
        
        # Layout berechnen
        pos = nx.spring_layout(G)
        
        # Zeichne Graphen
        plt.figure(figsize=(12, 8))
        
        # Knoten mit verschiedenen Farben basierend auf Status
        node_colors = {
            'completed': 'green',
            'available': 'yellow',
            'pending': 'lightgrey'
        }
        
        for status, color in node_colors.items():
            nodes = [node for node, data in G.nodes(data=True) if data['status'] == status]
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=color, alpha=0.8)
        
        # Kanten zeichnen
        edge_colors = {
            'predecessor': 'blue',
            'machine': 'red',
            'job': 'green'
        }
        
        for edge_type, color in edge_colors.items():
            edges = [(u, v) for u, v, data in G.edges(data=True) if data['type'] == edge_type]
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=color, alpha=0.6)
        
        # Labels hinzufügen
        labels = {node: f"{node}\n({data['machine']})" for node, data in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
        
        # Legende
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=status)
            for status, color in node_colors.items()
        ] + [
            Line2D([0], [0], color=color, lw=2, label=edge_type)
            for edge_type, color in edge_colors.items()
        ]
        
        plt.legend(handles=legend_elements, loc='upper right')
        plt.title('Job Scheduling Graph')
        plt.axis('off')
        plt.tight_layout()
        plt.show()


class GraphTransformer(nn.Module):
    """
    Graph Transformer Netzwerk für das Job-Scheduling-Problem
    """
    def __init__(self, node_features, edge_features, hidden_dim=64, num_heads=4):
        super(GraphTransformer, self).__init__()
        
        # Einbettung der Knotenfeatures
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        
        # Graph Attention Layers
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=num_heads, edge_dim=edge_features)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, edge_dim=edge_features)
        
        # Ausgabeschicht
        self.output = nn.Linear(hidden_dim, 1)
        
        # Dropout für Regularisierung
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, edge_index, edge_attr, batch=None):
        # Einbettung der Knoten
        x = self.node_embedding(x)
        x = F.relu(x)
        
        # Erste GAT-Schicht
        x = self.gat1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Zweite GAT-Schicht
        x = self.gat2(x, edge_index, edge_attr)
        x = F.relu(x)
        
        # Ausgabewerte für jeden Knoten
        node_values = self.output(x).squeeze(-1)
        
        return node_values


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) Agent
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64, num_heads=4):
        # Policy network (Actor)
        self.actor = GraphTransformer(state_dim, 3, hidden_dim, num_heads).to(device)
        
        # Value network (Critic)
        self.critic = GraphTransformer(state_dim, 3, hidden_dim, num_heads).to(device)

        
        # Optimierer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0001)  # Von 0.0003 auf 0.0001
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.0005)  # Von 0.001 auf 0.0005
        
        # PPO-Parameter
        self.clip_epsilon = 0.1  # Von 0.2 auf 0.1 reduziert
        self.gamma = 0.995       # Von 0.99 auf 0.995 erhöht
        self.lambda_gae = 0.97   # Von 0.95 auf 0.97 erhöht
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        
        # Speicher für Erfahrungen
        self.reset_memory()
    
    def reset_memory(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.masks = []
    
    def act(self, state):
        # Graph-Daten extrahieren
        x = state['x'].to(device)
        edge_index = state['edge_index'].to(device)
        edge_attr = state['edge_attr'].to(device)
        operation_mask = state['operation_mask'].to(device)
        
        # Actor: Berechnet Action-Logits für jeden Knoten
        with torch.no_grad():
            node_values = self.actor(x, edge_index, edge_attr)
            
            # Maske anwenden (nur verfügbare Operationen auswählen)
            masked_values = torch.where(
                operation_mask,
                node_values,
                torch.tensor(-1e9, device=device)
            )
            
            # Softmax über die maskierten Werte
            probs = F.softmax(masked_values, dim=0)
            
            # Wert für den Zustand schätzen
            state_value = self.critic(x, edge_index, edge_attr).mean().item()
            
            # Stochastische Auswahl einer Aktion
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            # Konvertiere Knoten-Index zu Operation-Queue-Index
            op_idx = action.item()
            queue_idx = None
            
            for i, op in enumerate(state['operation_queue']):
                if state['op_to_idx'][op['name']] == op_idx:
                    queue_idx = i
                    break
            
            # Falls kein entsprechender Index gefunden wurde, wähle eine zufällige Operation
            if queue_idx is None and state['operation_queue']:
                queue_idx = random.randrange(len(state['operation_queue']))
        
        return queue_idx, log_prob.item(), state_value
    
    def store_transition(self, state, action, log_prob, value, reward, done, mask):
        """
        Speichert eine Transition im Erfahrungsspeicher
        """
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.masks.append(mask)
    
    def compute_gae(self, next_value):
        """
        Berechnet Generalized Advantage Estimation (GAE)
        """
        values = self.values + [next_value]
        advantages = []
        gae = 0
        
        for i in reversed(range(len(self.rewards))):
            delta = self.rewards[i] + self.gamma * values[i+1] * (1 - self.dones[i]) - values[i]
            gae = delta + self.gamma * self.lambda_gae * (1 - self.dones[i]) * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def update(self, batch_size=64, epochs=10):
        """
        Aktualisiert das Policy- und Value-Netzwerk mit PPO
        """
        # Anzahl der Beispiele
        n_samples = len(self.states)
        
        # Zuletzt berechneten Wert für GAE verwenden
        next_value = 0  # Finale Zustände haben keinen Wert
        advantages = self.compute_gae(next_value)
        
        # Zurückgegebene Werte berechnen
        returns = [adv + val for adv, val in zip(advantages, self.values)]
        
        # Indizes für Batch-Erzeugung
        indices = np.arange(n_samples)
        
        # Batch-Durchläufe
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        for _ in range(epochs):
            np.random.shuffle(indices)
            
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                # Batch-Daten sammeln
                batch_states = [self.states[i] for i in batch_indices]
                batch_actions = [self.actions[i] for i in batch_indices]
                batch_log_probs = [self.log_probs[i] for i in batch_indices]
                batch_returns = [returns[i] for i in batch_indices]
                batch_advantages = [advantages[i] for i in batch_indices]
                
                # Vektorisierte Daten für alle Zustände im Batch
                batch_x = torch.cat([state['x'] for state in batch_states]).to(device)
                batch_edge_index = torch.cat([state['edge_index'] for state in batch_states], dim=1).to(device)
                batch_edge_attr = torch.cat([state['edge_attr'] for state in batch_states]).to(device)
                
                # Batch-Indizes für jedes Beispiel erstellen
                batch_idx = []
                node_offset = 0
                for state in batch_states:
                    n_nodes = state['x'].size(0)
                    batch_idx.extend([node_offset] * n_nodes)
                    node_offset += 1
                batch_idx = torch.tensor(batch_idx, device=device)
                
                # Policy-Netzwerk: neue Log-Probs berechnen
                new_log_probs = []
                entropy_sum = 0
                
                for i, (state, action) in enumerate(zip(batch_states, batch_actions)):
                    x = state['x'].to(device)
                    edge_index = state['edge_index'].to(device)
                    edge_attr = state['edge_attr'].to(device)
                    operation_mask = state['operation_mask'].to(device)
                    
                    # Actor-Ausgaben berechnen
                    node_values = self.actor(x, edge_index, edge_attr)
                    
                    # Maske anwenden
                    masked_values = torch.where(
                        operation_mask,
                        node_values,
                        torch.tensor(-1e9, device=device)
                    )
                    
                    # Neue Aktion-Verteilung
                    probs = F.softmax(masked_values, dim=0)
                    action_dist = torch.distributions.Categorical(probs)
                    
                    # Log-Prob für die ausgeführte Aktion
                    op_idx = None
                    for j, op in enumerate(state['operation_queue']):
                        if j == action:
                            op_idx = state['op_to_idx'][op['name']]
                            break
                    
                    if op_idx is not None:
                        log_prob = action_dist.log_prob(torch.tensor(op_idx, device=device))
                        new_log_probs.append(log_prob)
                        
                        # Entropie berechnen
                        entropy = action_dist.entropy()
                        entropy_sum += entropy
                
                # Neue Log-Probs in Tensor umwandeln
                if new_log_probs:
                    new_log_probs = torch.stack(new_log_probs)
                    old_log_probs = torch.tensor(batch_log_probs, device=device)
                    
                    # Ratio berechnen (Verhältnis neuer zu alter Policy)
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    
                    # Batch-Vorteile in Tensor umwandeln
                    advantages = torch.tensor(batch_advantages, device=device)
                    
                    # PPO-Clipping-Objective
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                    actor_loss = -torch.min(surr1, surr2).mean()
                    
                    # Entropy hinzufügen für mehr Exploration
                    entropy = entropy_sum / len(batch_indices)
                    actor_loss -= self.entropy_coef * entropy
                    
                    # Critic-Loss berechnen
                    value_preds = []
                    for state in batch_states:
                        x = state['x'].to(device)
                        edge_index = state['edge_index'].to(device)
                        edge_attr = state['edge_attr'].to(device)
                        
                        value_pred = self.critic(x, edge_index, edge_attr).mean()
                        value_preds.append(value_pred)
                    
                    if value_preds:
                        value_preds = torch.stack(value_preds).squeeze()
                        returns = torch.tensor(batch_returns, device=device).float()
                        
                        if value_preds.dim() == 0 and returns.dim() == 1:
                            value_preds = value_preds.unsqueeze(0)  # Convert scalar to [1]
                        elif returns.dim() == 0 and value_preds.dim() == 1:
                            returns = returns.unsqueeze(0)  # Convert scalar to [1]
                        
                        critic_loss = F.mse_loss(value_preds, returns)
                        
                        # Gesamtverlust berechnen
                        loss = actor_loss + self.value_coef * critic_loss
                        
                        # Optimierer aktualisieren
                        self.actor_optimizer.zero_grad()
                        self.critic_optimizer.zero_grad()
                        loss.backward()
                        
                        # Gradient Clipping anwenden
                        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                        
                        self.actor_optimizer.step()
                        self.critic_optimizer.step()
                        
                        # Statistiken sammeln
                        total_actor_loss += actor_loss.item()
                        total_critic_loss += critic_loss.item()
                        total_entropy += entropy.item()
        
        # Speicher zurücksetzen
        self.reset_memory()
        
        # Durchschnittliche Verluste zurückgeben
        n_batches = n_samples // batch_size + (1 if n_samples % batch_size != 0 else 0)
        avg_actor_loss = total_actor_loss / (epochs * n_batches)
        avg_critic_loss = total_critic_loss / (epochs * n_batches)
        avg_entropy = total_entropy / (epochs * n_batches)
        
        return {
            'actor_loss': avg_actor_loss,
            'critic_loss': avg_critic_loss,
            'entropy': avg_entropy
        }
    
    def save(self, path):
        """
        Speichert die Modelle
        """
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, path)
    
    def load(self, path):
        """
        Lädt die Modelle
        """
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])


def train_ppo_agent(env, agent, num_episodes=100, max_steps=1000, update_freq=20, checkpoint_dir="checkpoints"):
    """
    Trainiert einen PPO-Agenten auf der Job-Scheduling-Umgebung
    
    Args:
        env: Die JobSchedulingEnvironment
        agent: Der PPO-Agent
        num_episodes: Anzahl der Trainingsepisoden
        max_steps: Maximale Anzahl von Schritten pro Episode
        update_freq: Frequenz, mit der das Netzwerk aktualisiert wird
        checkpoint_dir: Verzeichnis zum Speichern der Checkpoints
    
    Returns:
        episode_rewards: Liste der Gesamtbelohnungen je Episode
        episode_makespans: Liste der Makespan-Werte je Episode
    """
    episode_rewards = []
    episode_makespans = []
    
    # Verzeichnis für Checkpoints erstellen, falls es nicht existiert
    import os
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Beste Makespan initialisieren
    best_makespan = float('inf')
    
    for episode in tqdm(range(num_episodes), desc="Training PPO"):
        state = env.reset()
        episode_reward = 0
        step_count = 0
        
        while step_count < max_steps:
            # Aktion auswählen
            action, log_prob, value = agent.act(state)
            
            # Aktion ausführen
            next_state, reward, done, info = env.step(action)
            
            # Maske für non-terminal Zustände (für GAE)
            mask = 1.0 - float(done)
            
            # Transition speichern
            agent.store_transition(state, action, log_prob, value, reward, done, mask)
            
            state = next_state
            episode_reward += reward
            step_count += 1
            
            # Netzwerk aktualisieren, wenn genug Erfahrungen gesammelt wurden
            if step_count % update_freq == 0 or done:
                agent.update()
            
            if done:
                break
        
        # Makespan speichern
        makespan = info.get("makespan", float('inf'))
        episode_makespans.append(makespan)
        episode_rewards.append(episode_reward)
        
        # Modell speichern, wenn es das beste bisher ist
        if makespan < best_makespan and makespan > 0:
            best_makespan = makespan
            checkpoint_path = os.path.join(checkpoint_dir, f"best_model_makespan_{makespan:.2f}.pt")
            agent.save(checkpoint_path)
            print(f"Neues bestes Modell gespeichert mit Makespan: {makespan:.2f}")
        
        # Regelmäßige Checkpoints (z.B. alle 50 Episoden)
        if (episode + 1) % 50 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_episode_{episode+1}.pt")
            agent.save(checkpoint_path)
            print(f"Checkpoint gespeichert nach Episode {episode+1}")
        
        # Status ausgeben
        if (episode + 1) % 10 == 0:
            avg_reward = sum(episode_rewards[-10:]) / 10
            avg_makespan = sum(episode_makespans[-10:]) / 10
            print(f"Episode {episode+1}, Avg Reward: {avg_reward:.2f}, Avg Makespan: {avg_makespan:.2f}")
    
    # Finales Modell speichern
    final_path = os.path.join(checkpoint_dir, "final_model.pt")
    agent.save(final_path)
    print(f"Finales Modell gespeichert unter: {final_path}")
    
    return episode_rewards, episode_makespans


def evaluate_method(env, method=None, agent=None, episodes=10):
    """
    Evaluiert eine Scheduling-Methode (Heuristik oder Agent)
    
    Args:
        env: Die JobSchedulingEnvironment
        method: Die zu verwendende Heuristik (FIFO, LIFO, SPT, Random)
        agent: Optional, ein PPO-Agent für RL-basiertes Scheduling
        episodes: Anzahl der Evaluierungsepisoden
    
    Returns:
        avg_makespan: Durchschnittlicher Makespan
        all_makespans: Liste aller Makespan-Werte
    """
    if method:
        env.heuristic = method
    
    all_makespans = []
    
    for _ in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            if agent:
                action, _, _ = agent.act(state)
                state, _, done, info = env.step(action)
            else:
                state, _, done, info = env.step()  # Verwende die Heuristik
        
        makespan = info["makespan"]
        all_makespans.append(makespan)
    
    avg_makespan = sum(all_makespans) / len(all_makespans)
    return avg_makespan, all_makespans


def generate_synthetic_data(config=None):
    """Generiert synthetische Produktionsdaten"""
    if config is None:
        config = {
            "n_jobs": 50,
            "min_ops": 1,
            "max_ops": 5,
            "machines": ["M1", "M2", "M3", "M4"],
            "materials": ["Material_A", "Material_B", "Material_C"],
            "tools": ["Kühlmittel", "Öl", "Werkzeug", "Schablone"]
        }
    
    jobs = []

    for i in range(1, config["n_jobs"] + 1):
        n_ops = random.randint(config["min_ops"], config["max_ops"])
        operations = []

        for j in range(1, n_ops + 1):
            operation = {
                "Name": f"Job_{i}_Op{j}",
                "benötigteZeit": random.randint(20, 60),
                "Maschine": random.choice(config["machines"]),
                "Vorgänger": [f"Job_{i}_Op{j-1}"] if j > 1 else None,
                "produziertesMaterial": random.choice(config["materials"])
            }

            # Zufällig zusätzliche Attribute hinzufügen
            if random.random() < 0.4:  # 40% Wahrscheinlichkeit für Hilfsmittel
                n_tools = random.randint(1, 3)
                operation["benötigteHilfsmittel"] = random.sample(config["tools"], min(n_tools, len(config["tools"])))
                operation["umruestkosten"] = random.randint(10, 40)

            if random.random() < 0.3:  # 30% Wahrscheinlichkeit für Zwischenlager
                operation["zwischenlager"] = {
                    "minVerweildauer": random.randint(10, 20),
                    "lagerkosten": random.randint(1, 5)
                }

            operations.append(operation)

        # Komplexere Abhängigkeiten hinzufügen (zufällig)
        if n_ops > 2 and random.random() < 0.3:
            # Füge eine zusätzliche Abhängigkeit hinzu
            op_idx = random.randint(3, n_ops) - 1  # Operation 3 oder höher
            from_idx = random.randint(1, op_idx - 1)  # Eine frühere Operation
            
            if operations[op_idx]["Vorgänger"] is None:
                operations[op_idx]["Vorgänger"] = []
            
            if isinstance(operations[op_idx]["Vorgänger"], list):
                if f"Job_{i}_Op{from_idx}" not in operations[op_idx]["Vorgänger"]:
                    operations[op_idx]["Vorgänger"].append(f"Job_{i}_Op{from_idx}")
            else:
                operations[op_idx]["Vorgänger"] = [operations[op_idx]["Vorgänger"], f"Job_{i}_Op{from_idx}"]

        job = {
            "Name": f"Job_{i}",
            "Priorität": random.randint(1, 10),
            "Operationen": operations
        }

        jobs.append(job)
    
    # Füge Maschinenumrüstzeiten hinzu
    maschinenUmrüstzeiten = {}
    for machine in config["machines"]:
        maschinenUmrüstzeiten[machine] = {
            "standardZeit": random.randint(10, 20),
            "materialWechsel": random.randint(15, 25)
        }

    generated_data = {
        "jobs": jobs,
        "maschinenUmrüstzeiten": maschinenUmrüstzeiten
    }
    
    print(f"✅ Synthetische Daten für {config['n_jobs']} Jobs generiert")
    return generated_data

def save_data(data, filename="production_data.json"):
    """Speichert die generierten Daten in eine JSON-Datei"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"✅ Daten gespeichert in: {filename}")
    return filename


# Am Ende der Datei füge diese erweiterte main-Funktion ein:

if __name__ == "__main__":
    # Pfad zur Datendatei (JSON)
    data_file = "production_data.json"
    model_path = "ppo_agent_model.pt"
    
    # Konfiguration für die Datengenerierung
    config = {
        "n_jobs": 50,
        "min_ops": 1,
        "max_ops": 5,
        "machines": ["M1", "M2", "M3", "M4"],
        "materials": ["Material_A", "Material_B", "Material_C"],
        "tools": ["Kühlmittel", "Öl", "Werkzeug", "Schablone"]
    }
    
    # Generiere synthetische Daten
    data = generate_synthetic_data(config)
    
    # Speichere Daten in Datei
    save_data(data, data_file)
    
    # Methoden vergleichen
    print("\n=== Vergleich der Scheduling-Methoden ===")
    
    # Heuristiken definieren
    heuristics = ["FIFO", "LIFO", "SPT", "Random"]
    
    # Ergebnisse speichern
    results = {}
    
    # Heuristiken evaluieren
    for heuristic in heuristics:
        print(f"Evaluating {heuristic}...")
        env = JobSchedulingEnvironment(data_file, heuristic=heuristic)
        avg_makespan, all_makespans = evaluate_method(env, method=heuristic)
        results[heuristic] = {
            "avg_makespan": avg_makespan,
            "all_makespans": all_makespans
        }
        print(f"{heuristic}: Avg Makespan = {avg_makespan:.2f}")
    
    # PPO-Agent trainieren
    print("\n=== Training des PPO-Agenten ===")
    env = JobSchedulingEnvironment(data_file)
    
    # Feature-Dimension bestimmen
    state = env.reset()
    state_dim = state['x'].shape[1]
    action_dim = len(state['operation_queue'])  # Dynamische Aktionsdimension
    
    # Agent erstellen
    agent = PPOAgent(state_dim, action_dim, hidden_dim=128, num_heads=8)  # Erhöhte Kapazität
    
    # Hyperparameter für Training
    num_episodes = 300  # Mehr Episoden für besseres Training
    update_freq = 10    # Häufigere Updates
    
    # Training
    print(f"Training für {num_episodes} Episoden...")
    rewards, makespans = train_ppo_agent(env, agent, num_episodes=num_episodes, update_freq=10, checkpoint_dir="model_checkpoints")
    
    # Modell speichern
    agent.save(model_path)
    print(f"Modell gespeichert unter: {model_path}")
    
    # Training Ergebnisse visualisieren
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(makespans)
    plt.title('Makespan über Training')
    plt.xlabel('Episode')
    plt.ylabel('Makespan')
    
    plt.subplot(1, 2, 2)
    plt.plot(rewards)
    plt.title('Reward über Training')
    plt.xlabel('Episode')
    plt.ylabel('Gesamtreward')
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.show()
    
    # Trainiertes Modell evaluieren
    print("\n=== Evaluierung des trainierten PPO-Agenten ===")
    avg_makespan, all_makespans = evaluate_method(env, agent=agent, episodes=20)
    results["PPO"] = {
        "avg_makespan": avg_makespan,
        "all_makespans": all_makespans
    }
    print(f"PPO Agent: Avg Makespan = {avg_makespan:.2f}")
    
    # Ein Beispiel visualisieren
    print("\n=== Visualisierung des finalen Schedules ===")
    env.reset()
    done = False
    while not done:
        action, _, _ = agent.act(env.state)
        _, _, done, _ = env.step(action)
    
    print("Final schedule with PPO agent:")
    env.render()
    
    # Ergebnisse vergleichen - Balkendiagramm
    labels = list(results.keys())
    makespans = [results[method]["avg_makespan"] for method in labels]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, makespans, color=['skyblue']*len(heuristics) + ['green'])
    plt.xlabel('Scheduling Method')
    plt.ylabel('Average Makespan')
    plt.title('Comparison of Scheduling Methods')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Hervorheben der besten Methode
    best_idx = np.argmin(makespans)
    bars[best_idx].set_color('orange')
    
    for i, v in enumerate(makespans):
        plt.text(i, v + 1, f"{v:.2f}", ha='center')
    
    plt.savefig('method_comparison.png')
    plt.tight_layout()
    plt.show()
    
    # Ausgabe der besten Methode
    best_method = min(results.items(), key=lambda x: x[1]["avg_makespan"])
    print(f"\nBeste Methode: {best_method[0]} mit durchschnittlichem Makespan: {best_method[1]['avg_makespan']:.2f}")
    
    # Verbesserung gegenüber FIFO berechnen
    fifo_makespan = results["FIFO"]["avg_makespan"]
    best_makespan = best_method[1]["avg_makespan"]
    improvement = (fifo_makespan - best_makespan) / fifo_makespan * 100
    
    print(f"Verbesserung gegenüber FIFO: {improvement:.2f}%")

# Funktion zum Laden und Weitertrainieren eines Modells
def load_and_continue_training(model_path, data_file, additional_episodes=100):
    """
    Lädt ein gespeichertes Modell und trainiert es weiter
    """
    env = JobSchedulingEnvironment(data_file)
    
    # Feature-Dimension bestimmen
    state = env.reset()
    state_dim = state['x'].shape[1]
    action_dim = len(state['operation_queue'])
    
    # Agent erstellen und Modell laden
    agent = PPOAgent(state_dim, action_dim)
    agent.load(model_path)
    print(f"Modell geladen von: {model_path}")
    
    # Weitertraining
    print(f"Weitertraining für {additional_episodes} Episoden...")
    rewards, makespans = train_ppo_agent(env, agent, num_episodes=additional_episodes)
    
    # Modell speichern
    agent.save(model_path)
    print(f"Verbessertes Modell gespeichert unter: {model_path}")
    
    return agent, rewards, makespans