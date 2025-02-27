import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
from datetime import datetime

from environments.job_env import JobSchedulingEnv
from agents.ppo_agent import PPOAgent

def evaluate_agent(env, agent, n_episodes=10):
    """
    Evaluiert einen trainierten Agenten
    
    Args:
        env: Die Umgebung
        agent: Der trainierte Agent
        n_episodes: Anzahl der Evaluierungsepisoden
    
    Returns:
        dict: Evaluierungsergebnisse
    """
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

def compare_strategies(env, agent=None, strategies=None):
    """
    Vergleicht verschiedene Scheduling-Strategien mit dem trainierten Agenten
    
    Args:
        env: Die Umgebung
        agent: Der trainierte Agent (optional)
        strategies: Liste von Strategien zum Vergleich
    
    Returns:
        dict: Vergleichsergebnisse
    """
    if strategies is None:
        strategies = ['FIFO', 'LIFO', 'SPT']
    
    results = {}
    
    # Baseline-Strategien evaluieren
    for strategy in strategies:
        env_copy = env  # In einer realen Implementierung sollte hier eine Kopie erstellt werden
        env_copy.reset()
        _, _, _, info = env_copy.step(strategy)
        results[strategy] = info.get('makespan', 0)
    
    # Agenten evaluieren, falls vorhanden
    if agent:
        agent_results = evaluate_agent(env, agent, n_episodes=5)
        results['PPO'] = agent_results['avg_makespan']
    
    # Ergebnisse ausgeben
    print("\nVergleich der Strategien:")
    for name, makespan in results.items():
        print(f"  {name}: Makespan = {makespan:.2f}")
    
    # Visualisierung
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values())
    plt.title('Vergleich der Scheduling-Strategien')
    plt.xlabel('Strategie')
    plt.ylabel('Makespan')
    plt.grid(True, axis='y')
    
    # Speichern der Grafik
    plots_dir = os.path.join(os.getcwd(), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(plots_dir, f"strategy_comparison_{timestamp}.png"))
    plt.show()
    
    return results

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
    
    # Speichern der Grafik
    plots_dir = os.path.join(os.getcwd(), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(plots_dir, f"schedule_{timestamp}.png"))
    plt.show()