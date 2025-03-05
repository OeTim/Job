import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
import gym
from gym import spaces
import random

class JobSchedulingEnv(gym.Env):
    """
    Job-Shop Scheduling Environment für Reinforcement Learning.
    
    Diese Umgebung repräsentiert ein Job-Shop Scheduling Problem als RL-Umgebung,
    wobei der Agent entscheiden muss, welche Operation als nächstes ausgeführt werden soll
    und welche Heuristik (FIFO, LIFO, SPT) dafür verwendet werden soll.
    """
    
    def __init__(self, disjunctive_graph: nx.DiGraph, max_steps: int = 1000):
        """
        Initialisiert die Job-Shop Scheduling Umgebung.
        
        Args:
            disjunctive_graph: Der disjunktive Graph, der das Scheduling-Problem repräsentiert
            max_steps: Maximale Anzahl an Schritten pro Episode
        """
        super(JobSchedulingEnv, self).__init__()
        
        # Speichere den ursprünglichen Graphen
        self.original_graph = disjunctive_graph.copy()
        
        # Aktueller Graph (wird während der Episode verändert)
        self.graph = None
        
        # Maximale Anzahl an Schritten
        self.max_steps = max_steps
        
        # Aktueller Schritt
        self.current_step = 0
        
        # Aktueller Zeitplan (Schedule)
        self.schedule = {}  # Operation -> (start_time, end_time)
        
        # Aktuelle Zeit
        self.current_time = 0
        
        # Maschinen-Zustand: Wann ist jede Maschine wieder verfügbar
        self.machine_available_time = {}
        
        # Bereits geplante Operationen
        self.scheduled_operations = set()
        
        # Operationen, die als nächstes geplant werden können
        self.eligible_operations = set()
        
        # Heuristiken
        self.heuristics = ["FIFO", "LIFO", "SPT"]
        
        # Aktionsraum: Wähle eine Heuristik
        self.action_space = spaces.Discrete(len(self.heuristics))
        
        # Beobachtungsraum: Zustand des Graphen und der Scheduling-Umgebung
        # Wir verwenden ein Dictionary-Space für komplexe Beobachtungen
        self.observation_space = spaces.Dict({
            # Anzahl der noch zu planenden Operationen
            'remaining_ops': spaces.Discrete(1000),
            
            # Durchschnittliche Bearbeitungszeit der anstehenden Operationen
            'avg_processing_time': spaces.Box(low=0, high=1000, shape=(1,), dtype=np.float32),
            
            # Maschinenauslastung (für jede Maschine)
            'machine_utilization': spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32),
            
            # Kritischer Pfad Länge
            'critical_path_length': spaces.Box(low=0, high=10000, shape=(1,), dtype=np.float32),
            
            # Anzahl der Maschinenkonflikte
            'machine_conflicts': spaces.Discrete(1000),
            
            # Aktuelle Zeit
            'current_time': spaces.Box(low=0, high=10000, shape=(1,), dtype=np.float32),
        })
        
        # Initialisiere die Umgebung
        self.reset()
    
    def reset(self):
        """
        Setzt die Umgebung zurück und gibt die initiale Beobachtung zurück.
        
        Returns:
            Initiale Beobachtung
        """
        # Kopiere den ursprünglichen Graphen
        self.graph = self.original_graph.copy()
        
        # Setze Zustandsvariablen zurück
        self.current_step = 0
        self.current_time = 0
        self.schedule = {}
        self.machine_available_time = {}
        self.scheduled_operations = set()
        self.eligible_operations = set()
        
        # Initialisiere die planbaren Operationen
        self._initialize_eligible_operations()
        
        # Debug-Ausgabe
        print(f"Reset: Graph hat {self.graph.number_of_nodes()} Knoten und {self.graph.number_of_edges()} Kanten")
        print(f"Reset: {len(self.eligible_operations)} planbare Operationen gefunden")
        if not self.eligible_operations:
            print("WARNUNG: Keine planbaren Operationen nach Reset!")
            # Zeige alle Operationsknoten und ihre Vorgänger
            for node, attrs in self.graph.nodes(data=True):
                if attrs.get('type') == 'operation':
                    predecessors = list(self.graph.predecessors(node))
                    print(f"Operation {node} hat {len(predecessors)} Vorgänger: {predecessors}")
        
        return self._get_observation()
    
    def _initialize_eligible_operations(self):
        """
        Initialisiert die Menge der planbaren Operationen.
        Eine Operation ist planbar, wenn alle ihre Vorgänger bereits geplant sind.
        """
        self.eligible_operations.clear()
        
        # Finde alle Operationsknoten
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get('type') == 'operation':
                # Prüfe, ob alle Vorgänger bereits geplant sind
                predecessors = list(self.graph.predecessors(node))
                
                # Eine Operation ist planbar, wenn sie keine Vorgänger hat oder
                # alle ihre Vorgänger vom Typ 'conjunctive' sind und bereits geplant wurden
                is_eligible = True
                
                for pred in predecessors:
                    # Prüfe nur konjunktive Kanten (Vorgängerbeziehungen)
                    edge_data = self.graph.get_edge_data(pred, node)
                    if edge_data and edge_data.get('type') == 'conjunctive':
                        # Wenn der Vorgänger eine Operation ist und noch nicht geplant wurde
                        if (self.graph.nodes[pred].get('type') == 'operation' and 
                            pred not in self.scheduled_operations):
                            is_eligible = False
                            break
                
                if is_eligible:
                    self.eligible_operations.add(node)
        
        # Debug-Ausgabe
        if not self.eligible_operations:
            print("WARNUNG: Keine planbaren Operationen gefunden!")
            # Prüfe, ob es START-Knoten gibt
            start_nodes = [n for n, attrs in self.graph.nodes(data=True) 
                          if attrs.get('type') == 'control' and attrs.get('control_type') == 'START']
            if start_nodes:
                print(f"START-Knoten gefunden: {start_nodes}")
                # Prüfe die direkten Nachfolger des START-Knotens
                for start in start_nodes:
                    successors = list(self.graph.successors(start))
                    print(f"Nachfolger von {start}: {successors}")
            else:
                print("Kein START-Knoten gefunden!")
    
    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """
        Führt einen Schritt in der Umgebung aus.
        
        Args:
            action: Index der zu verwendenden Heuristik (0=FIFO, 1=LIFO, 2=SPT)
            
        Returns:
            Tuple aus (Beobachtung, Belohnung, Episode beendet, Info)
        """
        # Erhöhe den Schrittzähler
        self.current_step += 1
        
        # Wähle die Heuristik basierend auf der Aktion
        heuristic = self.heuristics[action]
        
        # Wähle die nächste Operation basierend auf der Heuristik
        next_operation = self._select_operation_with_heuristic(heuristic)
        
        # Wenn keine Operation verfügbar ist, gehe zur nächsten Ereigniszeit
        if next_operation is None:
            self._advance_to_next_event()
            return self._get_observation(), 0, False, {}
        
        # Plane die ausgewählte Operation
        self._schedule_operation(next_operation)
        
        # Aktualisiere die planbaren Operationen
        self._update_eligible_operations()
        
        # Berechne die Belohnung
        reward = self._calculate_reward()
        
        # Prüfe, ob die Episode beendet ist
        done = self._is_done()
        
        # Gib die neue Beobachtung, Belohnung, etc. zurück
        return self._get_observation(), reward, done, {}
    
    def _select_operation_with_heuristic(self, heuristic: str) -> Optional[str]:
        """
        Wählt die nächste Operation basierend auf der angegebenen Heuristik.
        
        Args:
            heuristic: Die zu verwendende Heuristik (FIFO, LIFO, SPT)
            
        Returns:
            Die ausgewählte Operation oder None, wenn keine verfügbar ist
        """
        if not self.eligible_operations:
            return None
        
        eligible_list = list(self.eligible_operations)
        
        try:
            if heuristic == "FIFO":
                # First In, First Out: Wähle die Operation, die am längsten wartet
                # Wir verwenden hier die Knotennummer als Proxy für die Einfügereihenfolge
                return min(eligible_list, key=lambda op: int(op.split('_Op')[1]) if '_Op' in op else float('inf'))
                
            elif heuristic == "LIFO":
                # Last In, First Out: Wähle die zuletzt verfügbar gewordene Operation
                # Wir verwenden hier die Knotennummer als Proxy für die Einfügereihenfolge
                return max(eligible_list, key=lambda op: int(op.split('_Op')[1]) if '_Op' in op else 0)
                
            elif heuristic == "SPT":
                # Shortest Processing Time: Wähle die Operation mit der kürzesten Bearbeitungszeit
                return min(eligible_list, key=lambda op: self.graph.nodes[op].get('time', float('inf')))
        except Exception as e:
            print(f"Fehler bei der Auswahl der Operation mit Heuristik {heuristic}: {e}")
            print(f"Eligible operations: {eligible_list}")
            # Fallback: Zufällige Auswahl
            return random.choice(eligible_list)
        
        # Fallback: Zufällige Auswahl
        return random.choice(eligible_list)
    
    def _schedule_operation(self, operation: str) -> None:
        """
        Plant die angegebene Operation ein.
        
        Args:
            operation: Die einzuplanende Operation
        """
        # Hole Informationen über die Operation
        op_attrs = self.graph.nodes[operation]
        processing_time = op_attrs.get('time', 0)
        machine = op_attrs.get('machine')
        
        # Bestimme die frühestmögliche Startzeit basierend auf Vorgängern
        earliest_start = 0
        for pred in self.graph.predecessors(operation):
            if pred != 'START' and pred in self.schedule:
                pred_end_time = self.schedule[pred][1]
                earliest_start = max(earliest_start, pred_end_time)
        
        # Berücksichtige die Verfügbarkeit der Maschine
        machine_available = self.machine_available_time.get(machine, 0)
        start_time = max(earliest_start, machine_available)
        end_time = start_time + processing_time
        
        # Aktualisiere den Zeitplan
        self.schedule[operation] = (start_time, end_time)
        
        # Aktualisiere die Maschinenverfügbarkeit
        self.machine_available_time[machine] = end_time
        
        # Markiere die Operation als geplant
        self.scheduled_operations.add(operation)
        self.eligible_operations.remove(operation)
    
    def _update_eligible_operations(self) -> None:
        """
        Aktualisiert die Menge der planbaren Operationen.
        """
        # Prüfe für jede Operation, ob alle Vorgänger geplant sind
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get('type') == 'operation' and node not in self.scheduled_operations and node not in self.eligible_operations:
                predecessors = list(self.graph.predecessors(node))
                all_predecessors_scheduled = all(
                    pred == 'START' or pred in self.scheduled_operations 
                    for pred in predecessors
                )
                
                if all_predecessors_scheduled:
                    self.eligible_operations.add(node)
    
    def _advance_to_next_event(self):
        """
        Geht zur nächsten Ereigniszeit vor (wenn eine Operation fertig wird).
        Aktualisiert die planbaren Operationen.
        """
        # Finde die nächste Ereigniszeit (wenn eine Operation fertig wird)
        next_event_time = float('inf')
        for op, (start, end) in self.schedule.items():
            if end > self.current_time and end < next_event_time:
                next_event_time = end
        
        # Wenn keine weiteren Ereignisse gefunden wurden, aber noch Operationen übrig sind,
        # haben wir möglicherweise einen Deadlock
        if next_event_time == float('inf'):
            print(f"Geplante Operationen: {len(self.scheduled_operations)}")
            print(f"Verbleibende Operationen: {self.graph.number_of_nodes() - len(self.scheduled_operations) - 2}")  # -2 für START/END
            print(f"Planbare Operationen: {len(self.eligible_operations)}")
            
            # Versuche, den Deadlock zu beheben, indem eine zufällige Operation planbar gemacht wird
            unscheduled_ops = [n for n, attr in self.graph.nodes(data=True) 
                              if attr.get('type') == 'operation' and n not in self.scheduled_operations]
            
            if unscheduled_ops:
                # Wähle eine zufällige Operation und mache sie planbar
                op_to_free = random.choice(unscheduled_ops)
                print(f"Versuche Deadlock zu lösen, indem {op_to_free} planbar gemacht wird")
                
                # Entferne alle eingehenden konjunktiven Kanten zu dieser Operation
                predecessors = list(self.graph.predecessors(op_to_free))
                for pred in predecessors:
                    if self.graph.edges[pred, op_to_free].get('type') == 'conjunctive':
                        self.graph.remove_edge(pred, op_to_free)
                        print(f"  Entferne Abhängigkeit {pred} -> {op_to_free}")
                
                # Füge die Operation zu den planbaren Operationen hinzu
                self.eligible_operations.add(op_to_free)
                return
        
        return
    
    # Gehe zur nächsten Ereigniszeit vor
    self.current_time = next_event_time
    
    # Aktualisiere die planbaren Operationen
    self._update_eligible_operations()
    
    def _calculate_reward(self) -> float:
        """
        Berechnet die Belohnung für den aktuellen Zustand.
        
        Returns:
            Die Belohnung
        """
        # Berechne den Makespan (Gesamtdauer)
        makespan = max([end for _, (_, end) in self.schedule.items()], default=0)
        
        # Berechne die Maschinenauslastung
        total_processing_time = sum([end - start for (start, end) in self.schedule.values()])
        num_machines = len(self.machine_available_time)
        if num_machines > 0 and makespan > 0:
            utilization = total_processing_time / (num_machines * makespan)
        else:
            utilization = 0
        
        # Berechne die Anzahl der noch verbleibenden Operationen
        remaining_ops = len(self.graph.nodes()) - 2 - len(self.scheduled_operations)  # -2 für START und END
        
        # Belohnung basierend auf Fortschritt und Effizienz
        progress_reward = 1.0 if remaining_ops == 0 else 0.1  # Bonus für Fertigstellung
        utilization_reward = utilization * 0.5  # Belohnung für hohe Auslastung
        
        # Gesamtbelohnung
        reward = progress_reward + utilization_reward
        
        return reward
    
    def _is_done(self) -> bool:
        """
        Prüft, ob die Episode beendet ist.
        
        Returns:
            True, wenn alle Operationen geplant sind oder die maximale Schrittzahl erreicht ist
        """
        # Alle Operationen geplant?
        all_scheduled = all(
            attrs.get('type') != 'operation' or node in self.scheduled_operations
            for node, attrs in self.graph.nodes(data=True)
        )
        
        # Maximale Schrittzahl erreicht?
        max_steps_reached = self.current_step >= self.max_steps
        
        # Keine planbaren Operationen mehr und keine Ereignisse?
        deadlock = not self.eligible_operations and not self._has_future_events()
        
        # Wenn wir in einem Deadlock sind, geben wir eine Warnung aus
        if deadlock and not all_scheduled and not max_steps_reached:
            print(f"DEADLOCK erkannt nach {self.current_step} Schritten!")
            print(f"Geplante Operationen: {len(self.scheduled_operations)}")
            print(f"Verbleibende Operationen: {len(self.graph.nodes()) - 2 - len(self.scheduled_operations)}")
        
        return all_scheduled or max_steps_reached or deadlock
    
    def _has_future_events(self) -> bool:
        """
        Prüft, ob es zukünftige Ereignisse gibt.
        
        Returns:
            True, wenn es zukünftige Ereignisse gibt
        """
        for _, (_, end) in self.schedule.items():
            if end > self.current_time:
                return True
        return False
    
    def _get_observation(self) -> Dict:
        """
        Erstellt die Beobachtung für den aktuellen Zustand.
        
        Returns:
            Dictionary mit der Beobachtung
        """
        # Anzahl der noch zu planenden Operationen
        remaining_ops = len(self.graph.nodes()) - 2 - len(self.scheduled_operations)  # -2 für START und END
        
        # Durchschnittliche Bearbeitungszeit der anstehenden Operationen
        eligible_times = [self.graph.nodes[op].get('time', 0) for op in self.eligible_operations]
        avg_processing_time = np.mean(eligible_times) if eligible_times else 0
        
        # Maschinenauslastung
        machine_utilization = np.zeros(10)  # Annahme: maximal 10 Maschinen
        for i, machine in enumerate(sorted(self.machine_available_time.keys())):
            if i >= 10:
                break
            
            # Berechne die Auslastung als Verhältnis von Bearbeitungszeit zu Gesamtzeit
            total_time = max(1, self.current_time)
            busy_time = sum(
                min(self.current_time, end) - start 
                for op, (start, end) in self.schedule.items() 
                if self.graph.nodes[op].get('machine') == machine
            )
            machine_utilization[i] = busy_time / total_time
        
        # Berechne die Länge des kritischen Pfads
        critical_path_length = 0
        if self.schedule:
            critical_path_length = max(end for _, (_, end) in self.schedule.items())
        
        # Zähle die Maschinenkonflikte
        machine_conflicts = sum(
            1 for _, _, attr in self.graph.edges(data=True)
            if attr.get('type') == 'disjunctive' and 
            not (attr['machine'] in self.machine_available_time)
        )
        
        # Erstelle die Beobachtung
        observation = {
            'remaining_ops': remaining_ops,
            'avg_processing_time': np.array([avg_processing_time], dtype=np.float32),
            'machine_utilization': machine_utilization.astype(np.float32),
            'critical_path_length': np.array([critical_path_length], dtype=np.float32),
            'machine_conflicts': machine_conflicts,
            'current_time': np.array([self.current_time], dtype=np.float32)
        }
        
        return observation