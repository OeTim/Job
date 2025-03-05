import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def visualize_graph(G, conflict_edges=None, figsize=(12, 8)):
    """Visualisiert den Abhängigkeitsgraphen"""
    plt.figure(figsize=figsize)
    
    # Position der Knoten berechnen
    pos = nx.spring_layout(G, seed=42)
    
    # Knoten nach Job einfärben
    job_colors = {}
    color_map = []
    
    for node in G.nodes():
        job = G.nodes[node]["job"]
        if job not in job_colors:
            job_colors[job] = np.random.rand(3,)
        color_map.append(job_colors[job])
    
    # Knoten zeichnen
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color=color_map, alpha=0.8)
    
    # Abhängigkeitskanten zeichnen
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, d in G.edges(data=True) 
                                            if d.get('type') == 'precedence'],
                          width=1.5, edge_color='black', arrows=True)
    
    # Konfliktkanten zeichnen, falls vorhanden
    if conflict_edges:
        nx.draw_networkx_edges(G, pos, edgelist=conflict_edges,
                              width=1.0, edge_color='red', style='dashed', arrows=False)
    
    # Knotenbeschriftungen
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title("Abhängigkeitsgraph der Produktionsplanung")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_schedule(schedule, title="Job-Scheduling-Plan"):
    """
    Visualisiert einen Scheduling-Plan als Gantt-Diagramm
    
    Args:
        schedule: Liste von geplanten Operationen
        title: Titel des Diagramms
    """
    if not schedule:
        print("Kein Schedule zum Visualisieren vorhanden.")
        return
    
    # Daten für das Gantt-Diagramm vorbereiten
    machines = sorted(list(set([op['machine'] for op in schedule])))
    jobs = sorted(list(set([op['job'] for op in schedule])))
    
    # Farbzuordnung für Jobs
    colors = plt.cm.tab20(np.linspace(0, 1, len(jobs)))
    job_colors = {job: colors[i] for i, job in enumerate(jobs)}
    
    # Gantt-Diagramm erstellen
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Y-Achse: Maschinen
    y_ticks = list(range(len(machines)))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(machines)
    
    # Operationen zeichnen
    for op in schedule:
        machine_idx = machines.index(op['machine'])
        start_time = op['start']
        duration = op['end'] - op['start']
        
        # Rechteck für die Operation zeichnen
        ax.barh(machine_idx, duration, left=start_time, height=0.5, 
                color=job_colors[op['job']], alpha=0.8)
        
        # Beschriftung hinzufügen
        ax.text(start_time + duration/2, machine_idx, op['operation'], 
                ha='center', va='center', color='black', fontsize=8)
    
    # Legende für Jobs
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=job_colors[job], label=job) for job in jobs]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Diagramm-Beschriftung
    ax.set_xlabel('Zeit')
    ax.set_ylabel('Maschine')
    ax.set_title(title)
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()


def display_statistics(stats, title="Produktionsstatistiken"):
    """Zeigt Statistiken der Simulation an"""
    simulation_time = stats.get("makespan", 0)
    
    # Durchschnittliche Fertigstellungszeit
    avg_completion = sum(stats["job_completion_times"].values()) / len(stats["job_completion_times"]) if stats["job_completion_times"] else 0
    
    # Maschinenauslastung
    machine_util = {m: (t / simulation_time) * 100 for m, t in stats["machine_utilization"].items()} if simulation_time > 0 else {}
    
    # Werkzeugauslastung
    tool_util = {t: (u / simulation_time) * 100 for t, u in stats["tool_utilization"].items()} if simulation_time > 0 else {}
    
    # Durchschnittliche Wartezeit
    avg_waiting = sum(stats["waiting_times"]) / len(stats["waiting_times"]) if stats["waiting_times"] else 0
    
    # Ausgabe
    print(f"\n=== {title} ===")
    print(f"Makespan: {simulation_time:.2f} Zeiteinheiten")
    print(f"Durchschnittliche Fertigstellungszeit: {avg_completion:.2f} Zeiteinheiten")
    print(f"Durchschnittliche Wartezeit: {avg_waiting:.2f} Zeiteinheiten")
    
    print("\nMaschinenauslastung:")
    for machine, util in machine_util.items():
        print(f"  {machine}: {util:.2f}%")
    
    print("\nWerkzeugauslastung:")
    for tool, util in tool_util.items():
        print(f"  {tool}: {util:.2f}%")
    
    # Visualisierung
    plt.figure(figsize=(12, 6))
    
    # Fertigstellungszeiten
    plt.subplot(1, 2, 1)
    plt.bar(stats["job_completion_times"].keys(), stats["job_completion_times"].values())
    plt.title("Fertigstellungszeiten der Jobs")
    plt.xticks(rotation=90)
    plt.ylabel("Zeit")
    
    # Maschinenauslastung
    plt.subplot(1, 2, 2)
    plt.bar(machine_util.keys(), machine_util.values())
    plt.title("Maschinenauslastung")
    plt.ylabel("Auslastung (%)")
    
    plt.tight_layout()
    plt.show()


def plot_training_results(results, show_plots=True, save_plots=True):
    """Visualisiert die Trainingsergebnisse"""
    if not (show_plots or save_plots):
        return
        
    plt.figure(figsize=(15, 10))
    
    # Plot für Belohnungen
    plt.subplot(2, 1, 1)
    plt.plot(results['rewards'])
    plt.title('Belohnungen während des Trainings')
    plt.xlabel('Episode')
    plt.ylabel('Belohnung')
    plt.grid(True)
    
    # Plot für Makespan
    plt.subplot(2, 1, 2)
    plt.plot(results['makespans'])
    plt.title('Makespan während des Trainings')
    plt.xlabel('Episode')
    plt.ylabel('Makespan')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Speichern der Grafik
    if save_plots:
        plots_dir = os.path.join(os.getcwd(), 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(plots_dir, f"training_results_{timestamp}.png"))
    
    if show_plots:
        plt.show()
    else:
        plt.close()
