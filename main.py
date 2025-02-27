import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import json
import os
import simpy
import torch
import argparse
import copy  # Added import for copy module
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
# Import custom modules
from environments.job_env import JobSchedulingEnv
from agents.ppo_agent import PPOAgent

# Konfiguration für die Produktionsdaten
config = {
    "n_jobs": 50,           # Anzahl der Jobs
    "min_ops": 2,           # Minimale Operationen pro Job
    "max_ops": 5,           # Maximale Operationen pro Job
    "machines": ["M1", "M2", "M3", "M4"],
    "materials": ["Material_A", "Material_B", "Material_C"],
    "tools": ["Werkzeug", "Öl", "Kühlmittel", "Schablone"]
}

# Konfiguration für das RL-Training
rl_config = {
    # PPO-Hyperparameter
    "gamma": 0.99,              # Discount-Faktor
    "policy_clip": 0.2,         # PPO-Clipping-Parameter
    "n_epochs": 10,             # Anzahl der Epochen pro Update
    "gae_lambda": 0.95,         # GAE-Lambda-Parameter
    "batch_size": 64,           # Batch-Größe für Updates
    "lr": 0.0003,               # Lernrate
    "entropy_coef": 0.01,       # Entropie-Koeffizient für Exploration
    
    # Transformer-Parameter
    "d_model": 128,             # Dimension des Modells
    "nhead": 4,                 # Anzahl der Attention-Heads
    "num_layers": 2,            # Anzahl der Transformer-Layer
    "dim_feedforward": 512,     # Dimension des Feedforward-Netzwerks
    
    # Trainingsparameter
    "n_episodes": 1000,         # Anzahl der Trainingsepisoden
    "max_steps": 100,           # Maximale Schritte pro Episode
    "update_interval": 20,      # Intervall für Netzwerk-Updates
    "eval_interval": 50,        # Intervall für Evaluierungen
    "save_interval": 100,       # Intervall zum Speichern des Modells
}

# Globale Variablen für den Datenzustand
G = None                  # Graph
op_to_job = {}            # Zuordnung von Operationen zu Jobs
conflict_edges = []       # Maschinenkonflikte
data_json = None          # Gespeicherte Daten

def generate_synthetic_data(config):
    """Generiert synthetische Produktionsdaten"""
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
                n_tools = random.randint(1, 2)
                operation["benötigteHilfsmittel"] = random.sample(config["tools"], n_tools)
                operation["umruestzeit"] = random.randint(1, 10)
                operation["umruestkosten"] = random.randint(10, 50)

            if random.random() < 0.3:  # 30% Wahrscheinlichkeit für Zwischenlager
                operation["zwischenlager"] = {
                    "minVerweildauer": random.randint(5, 20),
                    "lagerkosten": random.randint(1, 5)
                }

            operations.append(operation)

        job = {
            "Name": f"Job_{i}",
            "Priorität": random.randint(1, 10),
            "Operationen": operations
        }

        jobs.append(job)

    generated_data = {"jobs": jobs}
    print(f"✅ Synthetische Daten für {config['n_jobs']} Jobs generiert")
    return generated_data

def save_data(data, filename="production_data.json"):
    """Speichert die generierten Daten in eine JSON-Datei"""
    filepath = os.path.join(os.getcwd(), filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"✅ Daten gespeichert in: {filepath}")
    return filepath

def load_data(filename="production_data.json"):
    """Lädt Daten aus einer JSON-Datei"""
    filepath = os.path.join(os.getcwd(), filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✅ Daten geladen aus: {filepath}")
    return data

def build_dependency_graph(data):
    """Erstellt einen Abhängigkeitsgraphen aus den Produktionsdaten"""
    global G, op_to_job, conflict_edges
    
    G = nx.DiGraph()
    op_to_job = {}
    conflict_edges = []
    
    # Knoten für alle Operationen erstellen
    for job in data["jobs"]:
        job_name = job["Name"]
        for op in job["Operationen"]:
            op_name = op["Name"]
            G.add_node(op_name, 
                       time=op["benötigteZeit"], 
                       machine=op["Maschine"],
                       job=job_name)
            op_to_job[op_name] = job_name
    
    # Abhängigkeitskanten hinzufügen
    for job in data["jobs"]:
        for op in job["Operationen"]:
            if op["Vorgänger"]:
                for pred in op["Vorgänger"]:
                    if pred in G:  # Prüfen, ob der Vorgänger existiert
                        G.add_edge(pred, op["Name"], type="precedence")
    
    # Maschinenkonflikte identifizieren
    machine_ops = {}
    for node in G.nodes():
        machine = G.nodes[node]["machine"]
        if machine not in machine_ops:
            machine_ops[machine] = []
        machine_ops[machine].append(node)
    
    # Konfliktkanten für Operationen auf der gleichen Maschine hinzufügen
    for machine, ops in machine_ops.items():
        for i in range(len(ops)):
            for j in range(i+1, len(ops)):
                if op_to_job[ops[i]] != op_to_job[ops[j]]:  # Nur zwischen verschiedenen Jobs
                    conflict_edges.append((ops[i], ops[j]))
    
    print(f"✅ Abhängigkeitsgraph erstellt mit {G.number_of_nodes()} Knoten und {G.number_of_edges()} Kanten")
    print(f"✅ {len(conflict_edges)} Maschinenkonflikte identifiziert")
    return G, conflict_edges

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

def simulate_production(data, strategy=None, random_seed=42):
    """
    Simuliert die Produktion mit SimPy
    
    Args:
        data: Produktionsdaten
        strategy: Scheduling-Strategie (None, 'FIFO', 'LIFO', 'SPT')
        random_seed: Seed für Zufallszahlen
    """
    random.seed(random_seed)
    env = simpy.Environment()
    
    # Ressourcen erstellen
    machines = {machine: simpy.Resource(env, capacity=1) for machine in config["machines"]}
    tools = {tool: simpy.Resource(env, capacity=3) for tool in config["tools"]}
    
    # Statistiken
    stats = {
        "job_completion_times": {},
        "machine_utilization": {machine: 0 for machine in machines},
        "tool_utilization": {tool: 0 for tool in tools},
        "waiting_times": [],
        "scheduled_operations": []  # Liste der geplanten Operationen für Gantt-Diagramm
    }
    
    # Prozess für einen Job
    def job_process(env, job):
        job_name = job["Name"]
        start_time = env.now
        
        for op in job["Operationen"]:
            op_name = op["Name"]
            machine_name = op["Maschine"]
            
            # Warten auf Maschine
            wait_start = env.now
            with machines[machine_name].request() as req:
                yield req
                wait_time = env.now - wait_start
                stats["waiting_times"].append(wait_time)
                
                # Werkzeuge anfordern, falls benötigt
                tool_reqs = []
                if "benötigteHilfsmittel" in op:
                    for tool in op["benötigteHilfsmittel"]:
                        tool_reqs.append(tools[tool].request())
                    
                    # Auf alle Werkzeuge warten
                    for req in tool_reqs:
                        yield req
                
                # Umrüstzeit, falls vorhanden
                if "umruestzeit" in op:
                    yield env.timeout(op["umruestzeit"])
                
                # Operation ausführen
                operation_start = env.now
                yield env.timeout(op["benötigteZeit"])
                operation_end = env.now
                
                # Operation zur Liste der geplanten Operationen hinzufügen
                stats["scheduled_operations"].append({
                    "job": job_name,
                    "operation": op_name,
                    "machine": machine_name,
                    "start": operation_start,
                    "end": operation_end
                })
                
                # Maschinennutzung aktualisieren
                stats["machine_utilization"][machine_name] += op["benötigteZeit"]
                
                # Werkzeuge freigeben
                for i, tool in enumerate(op.get("benötigteHilfsmittel", [])):
                    stats["tool_utilization"][tool] += op["benötigteZeit"]
                    tools[tool].release(tool_reqs[i])
                
                # Zwischenlagerung, falls erforderlich
                if "zwischenlager" in op:
                    yield env.timeout(op["zwischenlager"]["minVerweildauer"])
        
        # Job abgeschlossen
        completion_time = env.now - start_time
        stats["job_completion_times"][job_name] = completion_time
    
    # Jobs nach Strategie sortieren
    jobs = data["jobs"].copy()
    if strategy == 'FIFO':
        # FIFO: Keine Änderung der Reihenfolge
        pass
    elif strategy == 'LIFO':
        # LIFO: Umgekehrte Reihenfolge
        jobs.reverse()
    elif strategy == 'SPT':
        # SPT: Sortieren nach Gesamtbearbeitungszeit
        jobs.sort(key=lambda job: sum(op["benötigteZeit"] for op in job["Operationen"]))
    
    # Alle Jobs starten
    for job in jobs:
        env.process(job_process(env, job))
    
    # Simulation ausführen
    env.run()
    
    # Makespan berechnen
    makespan = max(stats["job_completion_times"].values()) if stats["job_completion_times"] else 0
    stats["makespan"] = makespan
    
    return stats

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

def train_rl_agent(data, config=None, rl_config=None, show_plots=False):
    """
    Trainiert einen RL-Agenten für die Job-Scheduling-Optimierung
    
    Args:
        data: Produktionsdaten
        config: Konfiguration für die Umgebung
        rl_config: Konfiguration für das RL-Training
        show_plots: Ob Plots während des Trainings angezeigt werden sollen
    
    Returns:
        agent: Der trainierte Agent
        results: Trainingsergebnisse
    """
    if config is None:
        config = {}
    
    if rl_config is None:
        rl_config = {}
    
    # Umgebung erstellen
    env = JobSchedulingEnv(data, config)
    
    # Agent erstellen
    agent = PPOAgent(env, rl_config)
    
    # Trainingskonfiguration
    n_episodes = rl_config.get('n_episodes', 1000)
    max_steps = rl_config.get('max_steps', 100)
    update_interval = rl_config.get('update_interval', 20)
    eval_interval = rl_config.get('eval_interval', 50)
    save_interval = rl_config.get('save_interval', 100)
    
    # Verzeichnis für Modelle erstellen
    models_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Tracking-Variablen
    all_rewards = []
    episode_rewards = []
    all_makespans = []
    best_makespan = float('inf')
    best_model_path = None
    no_improvement_count = 0
    
    # Daten-Augmentation für mehr Diversität
    augmented_data_sets = []
    for i in range(5):  # 5 verschiedene Datensätze
        aug_data = copy.deepcopy(data)
        # Zufällige Änderungen an den Bearbeitungszeiten
        for job in aug_data["jobs"]:
            for op in job["Operationen"]:
                # Variiere die Bearbeitungszeit um ±20%
                variation = 1.0 + random.uniform(-0.2, 0.2)
                op["benötigteZeit"] = max(1, int(op["benötigteZeit"] * variation))
        augmented_data_sets.append(aug_data)
    
    # Trainingsschleife
    print("\n=== Starte RL-Training ===")
    for episode in tqdm(range(1, n_episodes + 1), desc="Training"):
        # Alle 50 Episoden den Datensatz wechseln
        if episode % 50 == 1 and augmented_data_sets:
            data_idx = (episode // 50) % len(augmented_data_sets)
            env = JobSchedulingEnv(augmented_data_sets[data_idx], config)
            agent.env = env
            print(f"\nWechsel zu Datensatz {data_idx+1}")
        
        state = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done and steps < max_steps:
            # Aktion auswählen
            action, prob, val = agent.choose_action(state)
            
            # Aktion ausführen
            next_state, reward, done, info = env.step(action)
            
            # Erfahrung speichern
            agent.remember(state, action, prob, val, reward, done)
            
            # Zustand aktualisieren
            state = next_state
            episode_reward += reward
            steps += 1
            
            # Lernen, wenn genügend Erfahrungen gesammelt wurden
            if steps % update_interval == 0:
                agent.learn()
        
        # Episode abgeschlossen
        episode_rewards.append(episode_reward)
        all_rewards.append(episode_reward)
        
        # Makespan erfassen
        makespan = info.get('makespan', 0)
        all_makespans.append(makespan)
        
        # Bestes Modell speichern
        # In der train_rl_agent Funktion, beim Speichern des Modells:
        if makespan < best_makespan and makespan > 0:
            best_makespan = makespan
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Metadaten zum Modellnamen hinzufügen
            n_jobs = len(env.jobs)
            best_model_path = os.path.join(models_dir, f"best_model_{timestamp}_jobs_{n_jobs}_makespan_{int(best_makespan)}.pt")
            
            # Modell speichern
            agent.save_models(best_model_path)
            
            # Zusätzlich Metadaten speichern
            model_info = {
                'timestamp': timestamp,
                'n_jobs': n_jobs,
                'makespan': best_makespan,
                'config': {k: str(v) if isinstance(v, (type, torch.device)) else v 
                          for k, v in rl_config.items() if not callable(v)}
            }
            
            info_path = os.path.join(models_dir, f"model_info_{timestamp}.json")
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, ensure_ascii=False, indent=4)
            
            print(f"\nNeuer bester Makespan: {best_makespan:.2f} - Modell gespeichert in {best_model_path}")
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        # Exploration erhöhen, wenn keine Verbesserung
        if no_improvement_count > 50:
            agent.entropy_coef = min(0.5, agent.entropy_coef * 1.5)  # Begrenze auf 0.5
            print(f"\nKeine Verbesserung seit 50 Episoden. Erhöhe Exploration auf {agent.entropy_coef:.4f}")
            
            # Zusätzlich: Lernrate anpassen
            agent.lr *= 0.8
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] = agent.lr
            print(f"Lernrate reduziert auf {agent.lr:.6f}")
            
            # Umgebung zurücksetzen mit leicht veränderten Daten
            if episode > 200:
                perturbed_data = copy.deepcopy(data)
                for job in perturbed_data["jobs"]:
                    random.shuffle(job["Operationen"])  # Operationsreihenfolge ändern
                env = JobSchedulingEnv(perturbed_data, config)
                agent.env = env
                print("Umgebung mit veränderter Operationsreihenfolge zurückgesetzt")
            
            no_improvement_count = 0
        
        # Regelmäßige Evaluation
        if episode % eval_interval == 0:
            avg_reward = np.mean(episode_rewards[-eval_interval:])
            avg_makespan = np.mean(all_makespans[-eval_interval:])
            print(f"\nEpisode {episode}: Durchschnittliche Belohnung = {avg_reward:.2f}, Durchschnittlicher Makespan = {avg_makespan:.2f}")
            
            # Vergleich mit Baseline-Strategien nur anzeigen, wenn gewünscht
            if show_plots:
                compare_strategies(data)
        
        # Modell regelmäßig speichern
        if episode % save_interval == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(models_dir, f"model_episode_{episode}_{timestamp}.pt")
            agent.save_models(model_path)
            print(f"\nModell gespeichert in: {model_path}")
    
    # Trainingsergebnisse
    results = {
        'rewards': all_rewards,
        'makespans': all_makespans,
        'best_makespan': best_makespan,
        'best_model_path': best_model_path
    }
    
    # Ergebnisse visualisieren, nur wenn gewünscht
    if show_plots:
        plot_training_results(results)
    
    return agent, results

def evaluate_agent(data, agent, n_episodes=10):
    """
    Evaluiert einen trainierten Agenten
    
    Args:
        data: Produktionsdaten
        agent: Der trainierte Agent
        n_episodes: Anzahl der Evaluierungsepisoden
    
    Returns:
        dict: Evaluierungsergebnisse
    """
    env = agent.env
    makespans = []
    rewards = []
    schedules = []
    
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Aktion auswählen (ohne Exploration)
            with torch.no_grad():
                action, _, _ = agent.choose_action(state)
            
            # Aktion ausführen
            next_state, reward, done, info = env.step(action)
            
            # Zustand aktualisieren
            state = next_state
            episode_reward += reward
        
        # Episode abgeschlossen
        makespans.append(info.get('makespan', 0))
        rewards.append(episode_reward)
        schedules.append(env.scheduled_operations)
    
    # Ergebnisse
    results = {
        'avg_makespan': np.mean(makespans),
        'min_makespan': np.min(makespans),
        'max_makespan': np.max(makespans),
        'std_makespan': np.std(makespans),
        'avg_reward': np.mean(rewards),
        'best_schedule': schedules[np.argmin(makespans)]
    }
    
    print("\nEvaluierungsergebnisse:")
    print(f"  Durchschnittlicher Makespan: {results['avg_makespan']:.2f}")
    print(f"  Bester Makespan: {results['min_makespan']:.2f}")
    print(f"  Schlechtester Makespan: {results['max_makespan']:.2f}")
    print(f"  Standardabweichung: {results['std_makespan']:.2f}")
    print(f"  Durchschnittliche Belohnung: {results['avg_reward']:.2f}")
    
    return results

def compare_strategies(data, agent=None, strategies=None, show_plots=True):
    """
    Vergleicht verschiedene Scheduling-Strategien mit dem trainierten Agenten
    
    Args:
        data: Produktionsdaten
        agent: Der trainierte Agent (optional)
        strategies: Liste von Strategien zum Vergleich
        show_plots: Ob Plots angezeigt werden sollen
    
    Returns:
        dict: Vergleichsergebnisse
    """
    if strategies is None:
        strategies = ['FIFO', 'LIFO', 'SPT']
    
    results = {}
    
    # Baseline-Strategien evaluieren
    for strategy in strategies:
        stats = simulate_production(data, strategy=strategy)
        results[strategy] = stats.get('makespan', 0)
    
    # Agenten evaluieren, falls vorhanden
    if agent:
        agent_results = evaluate_agent(data, agent, n_episodes=5)
        results['PPO'] = agent_results['avg_makespan']
    
    # Ergebnisse ausgeben
    print("\nVergleich der Strategien:")
    for name, makespan in results.items():
        print(f"  {name}: Makespan = {makespan:.2f}")
    
    # Visualisierung nur wenn gewünscht
    if show_plots:
        plt.figure(figsize=(10, 6))
        plt.bar(results.keys(), results.values())
        plt.title('Vergleich der Scheduling-Strategien')
        plt.xlabel('Strategie')
        plt.ylabel('Makespan')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()
    
    return results

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

def main():
    """Hauptfunktion zum Ausführen des Programms"""
    global data_json
    
    parser = argparse.ArgumentParser(description='Job-Scheduling-Optimierung mit RL')
    parser.add_argument('--mode', type=str, default='simulate', choices=['simulate', 'train', 'evaluate', 'compare'],
                        help='Ausführungsmodus (simulate, train, evaluate, compare)')
    parser.add_argument('--strategy', type=str, default=None, choices=['FIFO', 'LIFO', 'SPT'],
                        help='Scheduling-Strategie für Simulation')
    parser.add_argument('--model', type=str, default=None,
                        help='Pfad zum vortrainierten Modell')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Anzahl der Trainingsepisoden')
    parser.add_argument('--data', type=str, default=None,
                        help='Pfad zur Datendatei')
    parser.add_argument('--no-plots', action='store_true',
                        help='Keine Plots während des Trainings anzeigen')
    
    # Neue Argumente für Datengenerierung
    parser.add_argument('--generate-data', action='store_true',
                        help='Nur Daten generieren und speichern')
    parser.add_argument('--n-jobs', type=int, default=50,
                        help='Anzahl der Jobs für die Datengenerierung')
    parser.add_argument('--output', type=str, default='production_data.json',
                        help='Ausgabedatei für generierte Daten')
    
    args = parser.parse_args()
    
    # Verzeichnisstruktur sicherstellen
    os.makedirs(os.path.join(os.getcwd(), 'data'), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(), 'models'), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(), 'results'), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(), 'plots'), exist_ok=True)
    
    # Nur Daten generieren, wenn --generate-data angegeben ist
    if args.generate_data:
        # Konfiguration anpassen
        custom_config = config.copy()
        custom_config['n_jobs'] = args.n_jobs
        
        # Daten generieren
        generated_data = generate_synthetic_data(custom_config)
        
        # Daten speichern
        save_data(generated_data, args.output)
        return
    
    # Daten generieren oder laden
    if args.data:
        data_json = load_data(args.data)
    else:
        data_json = generate_synthetic_data(config)
        save_data(data_json)
    
    # Rest der Funktion bleibt unverändert...
    # Graph erstellen
    G, conflicts = build_dependency_graph(data_json)
    
    # Modus auswählen
    if args.mode == 'simulate':
        # Simulation durchführen
        print("Starte Simulation...")
        stats = simulate_production(data_json, strategy=args.strategy)
        
        # Statistiken anzeigen
        display_statistics(stats)
        
        # Gantt-Diagramm anzeigen
        visualize_schedule(stats["scheduled_operations"], title=f"Job-Scheduling-Plan ({args.strategy or 'Standard'})")
        
        # Graph visualisieren
        visualize_graph(G, conflicts)
        
    elif args.mode == 'train':
        # RL-Konfiguration anpassen
        custom_rl_config = rl_config.copy()
        if args.episodes:
            custom_rl_config['n_episodes'] = args.episodes
        
        # RL-Agent trainieren
        print("Starte Training des RL-Agenten...")
        agent, results = train_rl_agent(data_json, config, custom_rl_config, show_plots=not args.no_plots)
        
        # Bestes Modell evaluieren
        print(f"\nBestes Modell: {results['best_model_path']}")
        print(f"Bester Makespan: {results['best_makespan']:.2f}")
        
        # Vergleich mit Baseline-Strategien
        compare_strategies(data_json, agent, show_plots=not args.no_plots)
        
    elif args.mode == 'evaluate':
        # Modell laden
        if not args.model:
            print("Fehler: Für die Evaluierung muss ein Modell angegeben werden (--model)")
            return
        
        # Umgebung erstellen
        env = JobSchedulingEnv(data_json, config)
        
        # Agent erstellen und Modell laden
        agent = PPOAgent(env, rl_config)
        agent.load_models(args.model)
        print(f"Modell geladen aus: {args.model}")
        
        # Agent evaluieren
        results = evaluate_agent(data_json, agent, n_episodes=10)
        
        # Besten Schedule visualisieren
        visualize_schedule(results['best_schedule'], title="Bester Schedule des RL-Agenten")
        
        # Vergleich mit Baseline-Strategien
        compare_strategies(data_json, agent)
        
    elif args.mode == 'compare':
        # Strategien vergleichen
        strategies = ['FIFO', 'LIFO', 'SPT']
        
        # Agent hinzufügen, falls Modell angegeben
        agent = None
        if args.model:
            env = JobSchedulingEnv(data_json, config)
            agent = PPOAgent(env, rl_config)
            agent.load_models(args.model)
            print(f"Modell geladen aus: {args.model}")
        
        # Vergleich durchführen
        results = compare_strategies(data_json, agent, strategies)
        
        # Für jede Strategie einen Schedule visualisieren
        for strategy in strategies:
            stats = simulate_production(data_json, strategy=strategy)
            visualize_schedule(stats["scheduled_operations"], title=f"Schedule mit {strategy}")
        
        # Wenn Agent vorhanden, auch dessen besten Schedule visualisieren
        if agent:
            agent_results = evaluate_agent(data_json, agent, n_episodes=5)
            visualize_schedule(agent_results['best_schedule'], title="Bester Schedule des RL-Agenten")
    
    print("\nProgramm erfolgreich beendet.")

if __name__ == "__main__":
    main()