import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Set, Optional
import os

def create_disjunctive_graph(jobs_data: Dict) -> nx.DiGraph:
    """
    Erstellt einen disjunktiven Graphen aus den Job-Daten.
    
    Args:
        jobs_data: Dictionary mit den Job-Daten
        
    Returns:
        NetworkX DiGraph-Objekt
    """
    G = nx.DiGraph()
    
    # Füge START und END Knoten hinzu
    G.add_node("START", type="control", control_type="START", time=0)
    G.add_node("END", type="control", control_type="END", time=0)
    
    # Sammle alle Operationen nach Maschinen
    machine_operations = {}  # Maschine -> Liste von Operationen
    
    # Füge alle Operationen als Knoten hinzu
    for job in jobs_data.get("jobs", []):
        job_name = job.get("Name")
        job_priority = job.get("Priorität", 1)
        
        # Speichere alle Operationen dieses Jobs für Zyklusprüfung
        job_operations = []
        
        for op in job.get("Operationen", []):
            op_name = op.get("Name")
            processing_time = op.get("benötigteZeit", 0)
            machine = op.get("Maschine")
            
            # Füge Operation als Knoten hinzu
            G.add_node(op_name, type="operation", job=job_name, machine=machine, 
                      time=processing_time, priority=job_priority)
            job_operations.append(op_name)
            
            # Sammle Operationen nach Maschinen
            if machine not in machine_operations:
                machine_operations[machine] = []
            machine_operations[machine].append(op_name)
            
            # Verbinde mit START, wenn keine Vorgänger
            predecessors = op.get("Vorgänger", [])
            if not predecessors:
                G.add_edge("START", op_name, type="conjunctive", weight=0)
                print(f"Verbinde START mit {op_name}")
            else:
                # Konvertiere einzelnen Vorgänger zu Liste
                if not isinstance(predecessors, list):
                    predecessors = [predecessors]
                
                # Verbinde mit Vorgängern
                for pred in predecessors:
                    if pred is not None:  # Überspringe None-Vorgänger
                        G.add_edge(pred, op_name, type="conjunctive", 
                                  weight=G.nodes.get(pred, {}).get("time", 0))
                        print(f"Verbinde {pred} mit {op_name}")
            
            # Verbinde mit END (wird später entfernt, wenn Nachfolger hinzugefügt werden)
            G.add_edge(op_name, "END", type="conjunctive", weight=processing_time)
    
    # Entferne direkte Verbindungen zu END, wenn die Operation Nachfolger hat
    for node in list(G.predecessors("END")):
        if node != "START":  # Überspringe den START-Knoten
            has_successors = False
            for _, successor in G.out_edges(node):
                if successor != "END" and G.edges[node, successor].get("type") == "conjunctive":
                    has_successors = True
                    break
            
            if has_successors:
                G.remove_edge(node, "END")
    
    # Füge Maschinenkonflikte als Kanten hinzu (disjunctive arcs)
    for machine, ops in machine_operations.items():
        # Für jedes Paar von Operationen auf der gleichen Maschine
        for i in range(len(ops)):
            for j in range(i+1, len(ops)):
                op1 = ops[i]
                op2 = ops[j]
                
                # Füge bidirektionale Kanten hinzu (Konflikt)
                G.add_edge(op1, op2, type="disjunctive", weight=G.nodes[op1]["time"], machine=machine)
                G.add_edge(op2, op1, type="disjunctive", weight=G.nodes[op2]["time"], machine=machine)
    
    # Debug-Ausgabe
    print(f"Graph erstellt mit {G.number_of_nodes()} Knoten und {G.number_of_edges()} Kanten")
    print(f"Operationsknoten: {len([n for n, attr in G.nodes(data=True) if attr.get('type') == 'operation'])}")
    print(f"Konjunktive Kanten: {len([e for e in G.edges if G.edges[e].get('type') == 'conjunctive'])}")
    print(f"Disjunktive Kanten: {len([e for e in G.edges if G.edges[e].get('type') == 'disjunctive'])}")
    
    # Prüfe, ob es Operationen ohne ausgehende konjunktive Kanten gibt (außer zu END)
    for node, attrs in G.nodes(data=True):
        if attrs.get('type') == 'operation':
            has_conj_successor = False
            for _, succ in G.out_edges(node):
                if succ != "END" and G.edges[node, succ].get("type") == "conjunctive":
                    has_conj_successor = True
                    break
            
            if not has_conj_successor:
                print(f"Operation {node} hat keine konjunktiven Nachfolger außer END")
                # Stelle sicher, dass diese Operation mit END verbunden ist
                if not G.has_edge(node, "END"):
                    G.add_edge(node, "END", type="conjunctive", weight=G.nodes[node]["time"])
                    print(f"  -> Verbindung zu END hinzugefügt")
    
    # Prüfe auf Zyklen in konjunktiven Kanten
    conj_edges = [(u, v) for u, v, attr in G.edges(data=True) if attr.get('type') == 'conjunctive']
    conj_graph = nx.DiGraph()
    conj_graph.add_nodes_from(G.nodes())
    conj_graph.add_edges_from(conj_edges)
    
    try:
        cycles = list(nx.simple_cycles(conj_graph))
        if cycles:
            print(f"WARNUNG: {len(cycles)} Zyklen in konjunktiven Kanten gefunden!")
            for cycle in cycles[:3]:
                print(f"  Zyklus: {' -> '.join(cycle)}")
                
            # Entferne Zyklen, indem die schwächste Kante entfernt wird
            for cycle in cycles:
                # Finde die Kante mit dem geringsten Gewicht
                min_weight = float('inf')
                edge_to_remove = None
                
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    if G.has_edge(u, v):
                        weight = G.edges[u, v].get('weight', 0)
                        if weight < min_weight:
                            min_weight = weight
                            edge_to_remove = (u, v)
                
                if edge_to_remove:
                    print(f"  Entferne Kante {edge_to_remove[0]} -> {edge_to_remove[1]} zur Auflösung des Zyklus")
                    G.remove_edge(edge_to_remove[0], edge_to_remove[1])
    except nx.NetworkXNoCycle:
        pass
    
    return G

def visualize_disjunctive_graph(G: nx.DiGraph, filename: Optional[str] = None, 
                               show_weights: bool = False, show_disjunctive: bool = True,
                               max_operations: int = 8, show_plot: bool = False) -> None:
    """
    Visualisiert den disjunktiven Graphen.
    
    Args:
        G: NetworkX DiGraph-Objekt
        filename: Pfad zum Speichern der Visualisierung (optional)
        show_weights: Ob Kantengewichte angezeigt werden sollen
        show_disjunctive: Ob disjunktive Kanten angezeigt werden sollen
        max_operations: Maximale Anzahl an Operationen, die visualisiert werden sollen
        show_plot: Ob der Graph interaktiv angezeigt werden soll
    """
    plt.figure(figsize=(12, 8))
    
    # Knoten nach Typ gruppieren
    control_nodes = [n for n, attr in G.nodes(data=True) if attr.get("type") == "control"]
    operation_nodes = [n for n, attr in G.nodes(data=True) if attr.get("type") == "operation"]
    
    # Erstelle einen Subgraphen nur mit Operations-Knoten (ohne START/END)
    nodes_to_keep = operation_nodes
    
    # Begrenze die Anzahl der Operationen für bessere Lesbarkeit
    if len(operation_nodes) > max_operations:
        # Wähle die ersten Jobs aus (z.B. Job_1, Job_2)
        job_prefixes = set()
        for node in operation_nodes:
            job_prefix = node.split("_Op")[0]
            job_prefixes.add(job_prefix)
        
        # Sortiere und wähle die ersten 2 Jobs
        selected_jobs = sorted(list(job_prefixes))[:2]
        
        # Filtere Operationen, die zu den ausgewählten Jobs gehören
        filtered_nodes = [n for n in operation_nodes if any(n.startswith(job) for job in selected_jobs)]
        
        # Begrenze auf maximal max_operations Knoten
        if len(filtered_nodes) > max_operations:
            filtered_nodes = filtered_nodes[:max_operations]
        
        nodes_to_keep = filtered_nodes
    
    # Erstelle einen Subgraphen mit den ausgewählten Knoten
    subG = G.subgraph(nodes_to_keep).copy()
    
    # Positioniere die Knoten mit einem übersichtlichen Layout
    try:
        # Versuche ein hierarchisches Layout für bessere Struktur
        pos = nx.kamada_kawai_layout(subG)
    except:
        # Fallback auf Spring-Layout
        pos = nx.spring_layout(subG, seed=42)
    
    # Kanten nach Typ gruppieren
    conjunctive_edges = [(u, v) for u, v, attr in subG.edges(data=True) 
                         if attr.get("type") == "conjunctive"]
    disjunctive_edges = [(u, v) for u, v, attr in subG.edges(data=True) 
                         if attr.get("type") == "disjunctive"]
    
    # Färbe Knoten nach Job für bessere Unterscheidung
    job_colors = {}
    for node in subG.nodes():
        job_name = subG.nodes[node].get("job", "")
        if job_name not in job_colors:
            # Generiere unterschiedliche Farben für verschiedene Jobs
            job_colors[job_name] = plt.cm.tab10(len(job_colors) % 10)
    
    # Zeichne Knoten nach Job gruppiert
    for job_name, color in job_colors.items():
        job_nodes = [n for n in subG.nodes() if subG.nodes[n].get("job") == job_name]
        nx.draw_networkx_nodes(subG, pos, nodelist=job_nodes, node_color=[color], 
                              node_size=700, alpha=0.8)
    
    # Zeichne Kanten mit unterschiedlichen Stilen
    nx.draw_networkx_edges(subG, pos, edgelist=conjunctive_edges, 
                          edge_color="black", arrows=True, width=2)
    
    if show_disjunctive and disjunctive_edges:
        nx.draw_networkx_edges(subG, pos, edgelist=disjunctive_edges, 
                              edge_color="red", style="dashed", 
                              arrows=True, alpha=0.6, width=1.5)
    
    # Zeichne Knotenbeschriftungen
    labels = {}
    for node in subG.nodes():
        machine = subG.nodes[node].get("machine", "")
        time = subG.nodes[node].get("time", 0)
        labels[node] = f"{node}\n({machine}, {time})"
    
    nx.draw_networkx_labels(subG, pos, labels=labels, font_size=9, font_weight='bold')
    
    # Zeichne Kantenbeschriftungen, wenn gewünscht
    if show_weights:
        edge_labels = {(u, v): subG[u][v].get("weight", "") for u, v in conjunctive_edges}
        nx.draw_networkx_edge_labels(subG, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title("Disjunktiver Graph - Operationen und Konflikte", fontsize=14)
    plt.axis("off")
    
    # Legende für Kanten
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, label='Vorgängerbeziehung'),
        Line2D([0], [0], color='red', lw=1.5, linestyle='--', label='Maschinenkonflikt')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Save the figure if a filename is provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    
    # Show the plot interactively if requested
    if show_plot:
        plt.show(block=True)  # block=True ensures the window stays open
    else:
        plt.close()  # Close the figure if not showing interactively

def get_critical_path(G: nx.DiGraph) -> Tuple[List[str], int]:
    """
    Berechnet den kritischen Pfad im Graphen, berücksichtigt nur conjunctive Kanten.
    
    Args:
        G: NetworkX DiGraph-Objekt
        
    Returns:
        Tuple mit kritischem Pfad (Liste von Knoten) und Gesamtlänge
    """
    # Erstelle einen neuen Graphen nur mit conjunctive Kanten
    H = nx.DiGraph()
    
    # Kopiere alle Knoten
    for node, attr in G.nodes(data=True):
        H.add_node(node, **attr)
    
    # Kopiere nur conjunctive Kanten
    for u, v, attr in G.edges(data=True):
        if attr.get("type") == "conjunctive":
            H.add_edge(u, v, **attr)
    
    try:
        path = nx.dag_longest_path(H, weight="weight")
        length = nx.dag_longest_path_length(H, weight="weight")
        return path, length
    except nx.NetworkXError:
        return [], 0

def get_machine_conflicts(G: nx.DiGraph) -> Dict[str, List[Tuple[str, str]]]:
    """
    Gibt alle Maschinenkonflikte im Graphen zurück.
    
    Args:
        G: NetworkX DiGraph-Objekt
        
    Returns:
        Dictionary mit Maschinen als Schlüssel und Listen von Konflikten als Werte
    """
    conflicts = {}
    
    for u, v, attr in G.edges(data=True):
        if attr.get("type") == "disjunctive":
            machine = attr.get("machine")
            if machine not in conflicts:
                conflicts[machine] = []
            conflicts[machine].append((u, v))
    
    return conflicts