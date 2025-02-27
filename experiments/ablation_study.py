import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import argparse
from datetime import datetime
from pathlib import Path
import sys

# Füge das Hauptverzeichnis zum Pfad hinzu
sys.path.append(str(Path(__file__).parent.parent))

from environments.job_env import JobSchedulingEnv
from agents.ppo_agent import PPOAgent
from evaluate import evaluate_agent, compare_strategies
from main import config, rl_config, load_data, generate_synthetic_data

def run_ablation_study(data, base_config, ablation_configs, n_episodes=500, eval_episodes=10):
    """
    Führt eine Ablationsstudie durch, indem verschiedene Konfigurationen verglichen werden
    
    Args:
        data: Produktionsdaten
        base_config: Basis-Konfiguration
        ablation_configs: Liste von (Name, Konfiguration)-Tupeln für die Ablation
        n_episodes: Anzahl der Trainingsepisoden
        eval_episodes: Anzahl der Evaluierungsepisoden
    """
    results = {}
    
    # Für jede Konfiguration
    for name, config in ablation_configs:
        print(f"\n=== Training mit Konfiguration: {name} ===")
        
        # Umgebung und Agent erstellen
        env = JobSchedulingEnv(data, {})
        agent = PPOAgent(env, config)
        
        # Training
        for episode in range(1, n_episodes + 1):
            state = env.reset()
            done = False
            steps = 0
            
            while not done and steps < config.get('max_steps', 100):
                # Aktion auswählen
                action, prob, val = agent.choose_action(state)
                
                # Aktion ausführen
                next_state, reward, done, info = env.step(action)
                
                # Erfahrung speichern
                agent.remember(state, action, prob, val, reward, done)
                
                # Zustand aktualisieren
                state = next_state
                steps += 1
                
                # Lernen, wenn genügend Erfahrungen gesammelt wurden
                if steps % config.get('update_interval', 20) == 0:
                    agent.learn()
            
            # Status ausgeben
            if episode % 50 == 0:
                print(f"Episode {episode}/{n_episodes}")
        
        # Evaluation
        print(f"\nEvaluiere Konfiguration: {name}")
        eval_results = evaluate_agent(env, agent, eval_episodes)
        
        # Baseline-Vergleich
        baseline_results = {}
        strategies = ['FIFO', 'LIFO', 'SPT']
        
        for strategy in strategies:
            env.reset()
            _, _, _, info = env.step(strategy)
            baseline_results[strategy] = info.get('makespan', 0)
        
        # Ergebnisse speichern
        results[name] = {
            'eval_results': eval_results,
            'baseline_results': baseline_results,
            'config': config
        }
    
    # Ergebnisse visualisieren
    visualize_ablation_results(results)
    
    # Ergebnisse speichern
    save_ablation_results(results)
    
    return results

def visualize_ablation_results(results):
    """Visualisiert die Ergebnisse der Ablationsstudie"""
    plt.figure(figsize=(12, 8))
    
    # Daten für das Diagramm vorbereiten
    model_names = list(results.keys())
    model_makespans = [results[name]['eval_results']['avg_makespan'] for name in model_names]
    
    # Baseline-Strategien (nehme die erste Konfiguration als Referenz)
    first_config = next(iter(results.values()))
    baseline_names = list(first_config['baseline_results'].keys())
    baseline_values = [first_config['baseline_results'][name] for name in baseline_names]
    
    # Alle Namen und Werte kombinieren
    all_names = baseline_names + model_names
    all_values = baseline_values + model_makespans
    
    # Farben für die Balken
    colors = ['lightgray'] * len(baseline_names) + ['skyblue'] * len(model_names)
    
    # Balkendiagramm erstellen
    bars = plt.bar(range(len(all_names)), all_values, color=colors)
    plt.xticks(range(len(all_names)), all_names, rotation=45, ha='right')
    plt.title('Vergleich der Konfigurationen')
    plt.ylabel('Makespan')
    plt.grid(True, axis='y')
    
    # Werte über den Balken anzeigen
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                 f'{height:.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Speichern der Grafik
    plots_dir = os.path.join(os.getcwd(), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(plots_dir, f"ablation_study_{timestamp}.png"))
    plt.show()

def save_ablation_results(results):
    """Speichert die Ergebnisse der Ablationsstudie"""
    results_dir = os.path.join(os.getcwd(), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Ergebnisse in JSON-Format konvertieren
    results_data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'configurations': {}
    }
    
    for name, data in results.items():
        # Schedules können nicht direkt in JSON gespeichert werden
        eval_results_copy = data['eval_results'].copy()
        if 'best_schedule' in eval_results_copy:
            # Vereinfachte Version des Schedules speichern
            eval_results_copy['best_schedule'] = [
                {k: v for k, v in op.items() if k != 'job_object'} 
                for op in eval_results_copy['best_schedule']
            ]
        
        results_data['configurations'][name] = {
            'eval_results': eval_results_copy,
            'baseline_results': data['baseline_results'],
            'config': {k: str(v) if isinstance(v, (type, torch.device)) else v 
                      for k, v in data['config'].items()}
        }
    
    # In Datei speichern
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(results_dir, f"ablation_study_{timestamp}.json")
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=4)
    
    print(f"\nAblationsstudie-Ergebnisse gespeichert in: {filename}")

def main():
    parser = argparse.ArgumentParser(description='Ablationsstudie für Job-Scheduling')
    parser.add_argument('--data', type=str, default=None,
                        help='Pfad zur Datendatei (optional)')
    parser.add_argument('--episodes', type=int, default=500,
                        help='Anzahl der Trainingsepisoden')
    parser.add_argument('--eval-episodes', type=int, default=10,
                        help='Anzahl der Evaluierungsepisoden')
    args = parser.parse_args()
    
    # Daten laden oder generieren
    if args.data:
        data = load_data(args.data)
    else:
        data = generate_synthetic_data(config)
    
    # Basis-Konfiguration
    base_config = rl_config.copy()
    
    # Ablations-Konfigurationen definieren
    ablation_configs = [
        ("Baseline", base_config.copy()),
        ("Ohne Entropie", {**base_config.copy(), "entropy_coef": 0.0}),
        ("Höhere Lernrate", {**base_config.copy(), "lr": 0.001}),
        ("Kleinere Batch-Größe", {**base_config.copy(), "batch_size": 32}),
        ("Mehr Transformer-Layer", {**base_config.copy(), "num_layers": 4}),
        ("Weniger Transformer-Layer", {**base_config.copy(), "num_layers": 1})
    ]
    
    # Ablationsstudie durchführen
    run_ablation_study(
        data, 
        base_config, 
        ablation_configs, 
        n_episodes=args.episodes, 
        eval_episodes=args.eval_episodes
    )

if __name__ == "__main__":
    main()