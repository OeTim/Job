import argparse
from datetime import datetime
import os
import networkx as nx
from pathlib import Path
import copy  # Add this import
import numpy as np  # Also add this as it's used in the code
import random  # Add this for random operations
import json  # Add this for JSON operations
import torch  # Add this for torch operations
from tqdm import tqdm  # Add this for progress bars

from src.utils.config import config, rl_config
from src.utils.data_generator import generate_synthetic_data, save_data, load_data
from src.visualization.visualizer import (visualize_graph, visualize_schedule, 
                                        display_statistics, plot_training_results)
from src.simulation.simulator import simulate_production, compare_strategies
from environments.job_env import JobSchedulingEnv
from agents.ppo_agent import PPOAgent

# Global variables for data state
G = None
op_to_job = {}
conflict_edges = []
data_json = None

def build_dependency_graph(data):
    """Creates a dependency graph from production data"""
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
        
        # Regelmäßige Evaluation
        if episode % eval_interval == 0:
            avg_reward = np.mean(episode_rewards[-eval_interval:])
            avg_makespan = np.mean(all_makespans[-eval_interval:])
            print(f"\nEpisode {episode}: Durchschnittliche Belohnung = {avg_reward:.2f}, Durchschnittlicher Makespan = {avg_makespan:.2f}")
            
            # Vergleich mit Baseline-Strategien nur anzeigen, wenn gewünscht
            if show_plots:
                # Fix: Pass both data and config to compare_strategies
                compare_strategies(data, config)

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



def main():
    """Main function to run the program"""
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
    
    # Graph erstellen
    G, conflicts = build_dependency_graph(data_json)
    
    # Modus auswählen
    if args.mode == 'simulate':
        # Simulation durchführen
        print("\n=== Simulation mit Zufallsstrategie ===")
        random_stats = []
        for i in range(10):  # 10 Durchläufe
            # Fix: Use data_json instead of data
            stats = simulate_production(data_json, strategy='RANDOM', random_seed=i)
            random_stats.append(stats['makespan'])
            print(f"Durchlauf {i+1}: Makespan = {stats['makespan']:.2f}")
        
        print(f"\nDurchschnittlicher Makespan (RANDOM): {np.mean(random_stats):.2f}")
        print(f"Minimaler Makespan (RANDOM): {np.min(random_stats):.2f}")
        print(f"Maximaler Makespan (RANDOM): {np.max(random_stats):.2f}")
        print(f"Standardabweichung (RANDOM): {np.std(random_stats):.2f}")
        # Fix: Use data_json instead of data
        stats = simulate_production(data_json, strategy=args.strategy)
        
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
        compare_strategies(data, config, include_rl=True, rl_agent=agent, n_runs=10)

        
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
        compare_strategies(data_json, config, include_rl=True, rl_agent=agent, n_runs=10)
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
        
        # Get total operations from data
        total_ops = sum(len(job["Operationen"]) for job in data_json["jobs"])
        print(f"\nGesamtanzahl Operationen im Datensatz: {total_ops}")
        
        # Vergleich durchführen
        results = compare_strategies(data_json, config, include_rl=(agent is not None), rl_agent=agent)
        
        # Für jede Strategie einen Schedule visualisieren
        for strategy in strategies:
            stats = simulate_production(data_json, config=config, strategy=strategy)
            # Ensure all operations are included in visualization
            all_operations = []
            for job in data_json["jobs"]:
                for op in job["Operationen"]:
                    # Check if operation is in scheduled_operations
                    scheduled_op = next((sop for sop in stats["scheduled_operations"] 
                                      if sop["job"] == job["Name"] and sop["operation"] == op["Name"]), None)
                    if scheduled_op:
                        all_operations.append(scheduled_op)
                    else:
                        # Add unscheduled operation with zero duration
                        all_operations.append({
                            "job": job["Name"],
                            "operation": op["Name"],
                            "machine": op["Maschine"],
                            "start": 0,
                            "end": 0
                        })
            
            visualize_schedule(all_operations, title=f"Schedule mit {strategy}")
        
        # Wenn Agent vorhanden, auch dessen Schedule mit allen Operationen visualisieren
        if agent:
            agent_results = evaluate_agent(data_json, agent, n_episodes=5)
            best_schedule = agent_results['best_schedule']
            
            # Zähle die Gesamtanzahl der Operationen in den Originaldaten
            total_ops_count = sum(len(job["Operationen"]) for job in data_json["jobs"])
            scheduled_ops_count = len(best_schedule)
            
            print(f"\nOperationsstatistik für RL-Agent:")
            print(f"  Geplante Operationen: {scheduled_ops_count}")
            print(f"  Gesamtanzahl möglicher Operationen: {total_ops_count}")
            
            # Stelle sicher, dass alle Operationen in der Visualisierung enthalten sind
            all_operations = []
            original_jobs = copy.deepcopy(data_json["jobs"])
            
            # Füge alle geplanten Operationen hinzu
            for op in best_schedule:
                all_operations.append(op)
            
            # Überprüfe, ob alle Operationen aus den Originaldaten enthalten sind
            for job in original_jobs:
                job_name = job["Name"]
                for op in job["Operationen"]:
                    op_name = op["Name"]
                    # Prüfe, ob die Operation bereits im Schedule enthalten ist
                    if not any(sched_op["job"] == job_name and sched_op["operation"] == op_name for sched_op in all_operations):
                        # Füge fehlende Operation hinzu
                        all_operations.append({
                            "job": job_name,
                            "operation": op_name,
                            "machine": op["Maschine"],
                            "start": 0,
                            "end": 0  # Setze Dauer auf 0 für nicht geplante Operationen
                        })
            
            visualize_schedule(all_operations, title="Vollständiger Schedule des RL-Agenten")
            
            # Entfernen Sie diese zweite Visualisierung, die nur die geplanten Operationen zeigt
            # visualize_schedule(best_schedule, title="Bester Schedule des RL-Agenten")
            
            # Count total operations in original data
            total_ops = sum(len(job["Operationen"]) for job in data_json["jobs"])
            
            print("\nVergleich der Operationen:")
            print(f"Gesamtanzahl möglicher Operationen: {total_ops}")
            print(f"Anzahl geplanter Operationen (RL-Agent): {len(best_schedule)}")
            
            visualize_schedule(best_schedule, title="Bester Schedule des RL-Agenten")
            
            # Berechne Maschinenauslastung für den RL-Agenten
            machine_times = {}
            for op in best_schedule:
                machine = op['machine']
                duration = op['end'] - op['start']
                if machine not in machine_times:
                    machine_times[machine] = 0
                machine_times[machine] += duration
            
            makespan = agent_results['min_makespan']
            machine_utilization = {m: time/makespan*100 for m, time in machine_times.items()}
            
            print(f"\nDetaillierte Statistiken für RL-Agent:")
            print(f"  Bester Makespan: {makespan:.2f}")
            print(f"  Durchschnittliche Maschinenauslastung: {np.mean(list(machine_utilization.values())):.2f}%")
    
    print("\nProgramm erfolgreich beendet.")

if __name__ == "__main__":
    main()