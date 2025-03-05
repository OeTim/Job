import os
import torch
import random
import networkx as nx
import matplotlib.pyplot as plt
from json_handler import load_or_create_jobs
from disjunctive_graph import create_disjunctive_graph, visualize_disjunctive_graph
from scheduling_env import JobSchedulingEnv
from graph_transformer import create_graph_transformer
from ppo_agent import PPOAgent

def run_heuristic(env, heuristic_name, heuristic_index=None, max_steps=5000):
    """Führt eine Episode mit einer bestimmten Heuristik aus."""
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    print(f"\n--- Starte Heuristik: {heuristic_name} ---")
    print(f"Initiale planbare Operationen: {len(env.eligible_operations)}")
    
    # Wenn keine planbaren Operationen vorhanden sind, ist etwas falsch
    if not env.eligible_operations:
        print("FEHLER: Keine planbaren Operationen zu Beginn!")
        return -1000, float('inf')
    
    while not done and steps < max_steps:
        action = heuristic_index if heuristic_index is not None else env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        state = next_state
        steps += 1
        
        # Prüfe auf Deadlock
        if len(env.eligible_operations) == 0 and state['remaining_ops'] > 0:
            print(f"DEADLOCK erkannt bei Schritt {steps}! Verbleibende Ops: {state['remaining_ops']}")
            # Versuche, den Deadlock zu beheben
            if not handle_deadlock(env):
                print("Konnte Deadlock nicht beheben, breche ab.")
                return -1000, float('inf')
        
        if steps % 10 == 0:
            print(f"Schritt {steps}: Verbleibende Ops: {state['remaining_ops']}, "
                  f"Zeit: {state['current_time'][0]:.1f}, Planbare Ops: {len(env.eligible_operations)}")
    
    if steps >= max_steps:
        print(f"WARNUNG: Maximale Schrittanzahl ({max_steps}) erreicht!")
        return -500, float('inf')
    
    makespan = max([end for _, (_, end) in env.schedule.items()], default=0)
    print(f"Makespan: {makespan:.2f}, Reward: {total_reward:.2f}")
    
    return total_reward, makespan

def handle_deadlock(env):
    """Versucht, einen Deadlock zu beheben, indem eine Operation planbar gemacht wird."""
    print("Versuche, Deadlock zu beheben...")
    
    # Finde alle nicht geplanten Operationen
    unscheduled_ops = [n for n, attr in env.graph.nodes(data=True) 
                      if attr.get('type') == 'operation' and n not in env.scheduled_operations]
    
    if not unscheduled_ops:
        return False
    
    # Sortiere nach Priorität und wähle die mit höchster Priorität
    unscheduled_ops.sort(key=lambda n: env.graph.nodes[n].get('priority', 0), reverse=True)
    op_to_free = unscheduled_ops[0]
    
    print(f"Mache Operation {op_to_free} planbar")
    
    # Entferne alle eingehenden konjunktiven Kanten
    predecessors = list(env.graph.predecessors(op_to_free))
    for pred in predecessors:
        if env.graph.has_edge(pred, op_to_free) and env.graph.edges[pred, op_to_free].get('type') == 'conjunctive':
            env.graph.remove_edge(pred, op_to_free)
            print(f"  Entferne Abhängigkeit {pred} -> {op_to_free}")
    
    # Füge die Operation zu den planbaren Operationen hinzu
    env.eligible_operations.add(op_to_free)
    
    return True

def validate_graph(graph):
    """Validiert den disjunktiven Graphen auf häufige Probleme."""
    print("\n=== Validiere disjunktiven Graphen ===")
    
    # Prüfe auf isolierte Knoten
    isolated_nodes = list(nx.isolates(graph))
    if isolated_nodes:
        print(f"WARNUNG: {len(isolated_nodes)} isolierte Knoten gefunden: {isolated_nodes}")
    else:
        print("Keine isolierten Knoten gefunden.")
    
    # Prüfe auf Zyklen in konjunktiven Kanten (würde zu Deadlocks führen)
    conj_edges = [(u, v) for u, v, attr in graph.edges(data=True) if attr.get('type') == 'conjunctive']
    conj_graph = nx.DiGraph()
    conj_graph.add_nodes_from(graph.nodes())
    conj_graph.add_edges_from(conj_edges)
    
    cycles = []
    try:
        cycles = list(nx.simple_cycles(conj_graph))
        if cycles:
            print(f"WARNUNG: {len(cycles)} Zyklen in konjunktiven Kanten gefunden!")
            for cycle in cycles[:3]:  # Zeige nur die ersten 3 Zyklen
                print(f"  Zyklus: {' -> '.join(cycle)}")
        else:
            print("Keine Zyklen in konjunktiven Kanten gefunden.")
    except nx.NetworkXNoCycle:
        print("Keine Zyklen in konjunktiven Kanten gefunden.")
    
    # Prüfe auf fehlende Verbindungen zum START/END
    operation_nodes = [n for n, attr in graph.nodes(data=True) if attr.get('type') == 'operation']
    
    # Prüfe, ob alle Operationen erreichbar sind
    unreachable = set()
    if "START" in graph.nodes():
        reachable = set(nx.descendants(graph, "START"))
        unreachable = set(operation_nodes) - reachable
        if unreachable:
            print(f"WARNUNG: {len(unreachable)} Operationen sind vom START-Knoten nicht erreichbar!")
            print(f"  Nicht erreichbar: {list(unreachable)[:5]}...")
        else:
            print("Alle Operationen sind vom START-Knoten erreichbar.")
    
    # Prüfe, ob alle Operationen zum END führen
    cant_reach_end = set()
    if "END" in graph.nodes():
        can_reach_end = set()
        for node in operation_nodes:
            if nx.has_path(graph, node, "END"):
                can_reach_end.add(node)
        
        cant_reach_end = set(operation_nodes) - can_reach_end
        if cant_reach_end:
            print(f"WARNUNG: {len(cant_reach_end)} Operationen können den END-Knoten nicht erreichen!")
            print(f"  Können END nicht erreichen: {list(cant_reach_end)[:5]}...")
        else:
            print("Alle Operationen können den END-Knoten erreichen.")
    
    # Prüfe auf Deadlocks durch Maschinenkonflikte
    machine_conflicts = {}
    for u, v, attr in graph.edges(data=True):
        if attr.get('type') == 'disjunctive':
            machine = attr.get('machine')
            if machine not in machine_conflicts:
                machine_conflicts[machine] = []
            machine_conflicts[machine].append((u, v))
    
    print(f"Maschinenkonflikte: {len(machine_conflicts)} Maschinen mit insgesamt {sum(len(conflicts) for conflicts in machine_conflicts.values())} Konflikten")
    
    # Return True wenn keine Probleme gefunden wurden
    has_problems = bool(isolated_nodes or cycles or unreachable or cant_reach_end)
    
    return not has_problems

def fix_graph_issues(graph):
    """Behebt erkannte Probleme im Graphen."""
    print("\n=== Behebe Probleme im Graphen ===")
    
    fixed_issues = 0
    
    # 1. Verbinde isolierte Knoten mit START und END
    isolated_nodes = list(nx.isolates(graph))
    for node in isolated_nodes:
        if graph.nodes[node].get('type') == 'operation':
            print(f"Verbinde isolierten Knoten {node} mit START und END")
            graph.add_edge("START", node, type="conjunctive", weight=0)
            graph.add_edge(node, "END", type="conjunctive", weight=graph.nodes[node].get("time", 0))
            fixed_issues += 1
    
    # 2. Entferne Zyklen in konjunktiven Kanten
    conj_edges = [(u, v) for u, v, attr in graph.edges(data=True) if attr.get('type') == 'conjunctive']
    conj_graph = nx.DiGraph()
    conj_graph.add_nodes_from(graph.nodes())
    conj_graph.add_edges_from(conj_edges)
    
    try:
        cycles = list(nx.simple_cycles(conj_graph))
        for cycle in cycles:
            # Finde die Kante mit dem geringsten Gewicht im Zyklus
            min_weight = float('inf')
            edge_to_remove = None
            
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                if conj_graph.has_edge(u, v):
                    weight = graph.edges[u, v].get('weight', 0)
                    if weight < min_weight:
                        min_weight = weight
                        edge_to_remove = (u, v)
            
            if edge_to_remove:
                print(f"Entferne Kante {edge_to_remove[0]} -> {edge_to_remove[1]} zur Auflösung eines Zyklus")
                graph.remove_edge(edge_to_remove[0], edge_to_remove[1])
                fixed_issues += 1
    except nx.NetworkXNoCycle:
        pass
    
    # 3. Verbinde nicht erreichbare Operationen mit START
    operation_nodes = [n for n, attr in graph.nodes(data=True) if attr.get('type') == 'operation']
    if "START" in graph.nodes():
        reachable = set(nx.descendants(graph, "START"))
        unreachable = set(operation_nodes) - reachable
        
        for node in unreachable:
            print(f"Verbinde nicht erreichbaren Knoten {node} mit START")
            graph.add_edge("START", node, type="conjunctive", weight=0)
            fixed_issues += 1
    
    # 4. Verbinde Operationen, die END nicht erreichen können, mit END
    if "END" in graph.nodes():
        cant_reach_end = set()
        for node in operation_nodes:
            if not nx.has_path(graph, node, "END"):
                cant_reach_end.add(node)
        
        for node in cant_reach_end:
            print(f"Verbinde Knoten {node} mit END")
            graph.add_edge(node, "END", type="conjunctive", weight=graph.nodes[node].get("time", 0))
            fixed_issues += 1
    
    print(f"Insgesamt {fixed_issues} Probleme behoben")
    return fixed_issues > 0

def train_and_evaluate_transformer(env, graph, num_episodes=1000):
    """Trainiert und evaluiert den Graph Transformer mit PPO."""
    print("\n=== Starte Training des Graph Transformers ===")
    
    # Initialisiere PPO-Agent
    agent = PPOAgent(env, graph)
    
    # Training
    agent.train(num_episodes=num_episodes)
    
    # Evaluation
    print("\n--- Evaluiere trainierten Transformer ---")
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < 5000:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        steps += 1
        
        # Prüfe auf Deadlock
        if len(env.eligible_operations) == 0 and state['remaining_ops'] > 0:
            print(f"DEADLOCK erkannt bei Schritt {steps}! Verbleibende Ops: {state['remaining_ops']}")
            # Versuche, den Deadlock zu beheben
            if not handle_deadlock(env):
                print("Konnte Deadlock nicht beheben, breche ab.")
                return -1000, float('inf')
        
        if steps % 10 == 0:
            print(f"Schritt {steps}: Verbleibende Ops: {state['remaining_ops']}, "
                  f"Zeit: {state['current_time'][0]:.1f}")
    
    if steps >= 5000:
        print("WARNUNG: Maximale Schrittanzahl erreicht!")
        return -500, float('inf')
    
    makespan = max([end for _, (_, end) in env.schedule.items()], default=0)
    print(f"Makespan: {makespan:.2f}, Reward: {total_reward:.2f}")
    
    return total_reward, makespan

def main():
    # 1. Lade oder erstelle Job-Daten
    base_dir = os.path.dirname(__file__)
    json_path = os.path.join(base_dir, "jobs.json")
    jobs_data = load_or_create_jobs(json_path, num_jobs=20)
    print(f"Jobs geladen: {len(jobs_data['jobs'])}")
    
    # 2. Erstelle disjunktiven Graphen
    graph = create_disjunctive_graph(jobs_data)
    print(f"Graph erstellt: {graph.number_of_nodes()} Knoten, {graph.number_of_edges()} Kanten")
    
    # 2.1 Validiere den Graphen
    is_valid = validate_graph(graph)
    if not is_valid:
        print("WARNUNG: Der Graph enthält Probleme, die zu Deadlocks führen könnten.")
        print("Versuche, Probleme automatisch zu beheben...")
        
        # Behebe erkannte Probleme
        fixed = fix_graph_issues(graph)
        
        if fixed:
            print("Probleme wurden behoben. Validiere erneut...")
            is_valid = validate_graph(graph)
            if not is_valid:
                print("Es bestehen weiterhin Probleme im Graphen.")
        
        # Visualisiere den Graphen für Debugging
        visualize_disjunctive_graph(graph, filename="fixed_graph.png", show_plot=False)
        print("Graph wurde als 'fixed_graph.png' gespeichert.")
    
    # 3. Initialisiere Umgebung
    env = JobSchedulingEnv(graph)
    print("Umgebung initialisiert")
    
    # 4. Evaluiere Baseline-Heuristiken
    results = {}
    heuristics = {
        "FIFO": 0,
        "LIFO": 1,
        "SPT": 2,
        "RANDOM": None
    }
    
    for name, index in heuristics.items():
        try:
            results[name] = run_heuristic(env, name, index)
            print(f"Heuristik {name} erfolgreich ausgeführt")
        except Exception as e:
            print(f"Fehler bei Heuristik {name}: {str(e)}")
            results[name] = (-1000, float('inf'))  # Schlechtes Ergebnis für fehlgeschlagene Heuristik
    
    # 5. Trainiere und evaluiere Graph Transformer
    try:
        results["TRANSFORMER"] = train_and_evaluate_transformer(env, graph, num_episodes=1000)
        print("Transformer erfolgreich trainiert und evaluiert")
    except Exception as e:
        print(f"Fehler beim Training des Transformers: {str(e)}")
        results["TRANSFORMER"] = (-1000, float('inf'))
    
    # 6. Vergleiche Ergebnisse
    print("\n=== Vergleich aller Methoden ===")
    print("Methode        | Makespan  | Reward")
    print("---------------|-----------|--------")
    
    # Sortiere nach Makespan (aufsteigend)
    sorted_results = sorted(results.items(), key=lambda x: x[1][1])
    
    for name, (reward, makespan) in sorted_results:
        if makespan == float('inf'):
            print(f"{name:<13} | {'FEHLER':>9} | {reward:8.2f}")
        else:
            print(f"{name:<13} | {makespan:9.2f} | {reward:8.2f}")
    
    # 7. Analysiere Verbesserung gegenüber LIFO
    if results["LIFO"][1] != float('inf') and results["TRANSFORMER"][1] != float('inf'):
        lifo_makespan = results["LIFO"][1]
        transformer_makespan = results["TRANSFORMER"][1]
        improvement = (lifo_makespan - transformer_makespan) / lifo_makespan * 100
    
    print(f"\nVerbesserung gegenüber LIFO: {improvement:.2f}%")

if __name__ == "__main__":
    main()