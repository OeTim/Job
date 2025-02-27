import os
import numpy as np
import torch
import json
import itertools
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from environments.job_env import JobSchedulingEnv
from agents.ppo_agent import PPOAgent

def grid_search(data, base_config, param_grid, n_episodes=300, n_eval_episodes=5):
    """
    Führt eine Grid-Search für Hyperparameter durch
    
    Args:
        data: Produktionsdaten
        base_config: Basis-Konfiguration
        param_grid: Dictionary mit zu testenden Parametern
        n_episodes: Anzahl der Trainingsepisoden pro Konfiguration
        n_eval_episodes: Anzahl der Evaluierungsepisoden
    
    Returns:
        dict: Ergebnisse der Grid-Search
    """
    # Parameter-Kombinationen erstellen
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    results = []
    best_makespan = float('inf')
    best_config = None
    
    print(f"Starte Grid-Search mit {len(param_combinations)} Konfigurationen")
    
    for i, combination in enumerate(param_combinations):
        # Konfiguration erstellen
        config = base_config.copy()
        for name, value in zip(param_names, combination):
            config[name] = value
        
        print(f"\n=== Konfiguration {i+1}/{len(param_combinations)} ===")
        print("Parameter:")
        for name, value in zip(param_names, combination):
            print(f"  {name}: {value}")
        
        # Umgebung und Agent erstellen
        env = JobSchedulingEnv(data, {})
        agent = PPOAgent(env, config)
        
        # Training
        print(f"Training für {n_episodes} Episoden...")
        train_results = train_with_config(env, agent, config, n_episodes)
        
        # Evaluation
        print("Evaluiere trainiertes Modell...")
        eval_results = evaluate_with_config(env, agent, n_eval_episodes)
        
        # Ergebnisse speichern
        config_results = {
            'config': {name: value for name, value in zip(param_names, combination)},
            'train_results': train_results,
            'eval_results': eval_results
        }
        results.append(config_results)
        
        # Bestes Modell aktualisieren
        if eval_results['avg_makespan'] < best_makespan:
            best_makespan = eval_results['avg_makespan']
            best_config = config.copy()
            print(f"\n✅ Neue beste Konfiguration gefunden! Makespan: {best_makespan:.2f}")
    
    # Ergebnisse speichern
    save_tuning_results(results, best_config)
    
    # Ergebnisse visualisieren
    visualize_tuning_results(results, param_names)
    
    return results, best_config

def train_with_config(env, agent, config, n_episodes):
    """Trainiert einen Agenten mit der gegebenen Konfiguration"""
    # Tracking-Variablen
    all_rewards = []
    all_makespans = []
    
    # Trainingsschleife
    for episode in tqdm(range(1, n_episodes + 1), desc="Training"):
        state = env.reset()
        done = False
        episode_reward = 0
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
            episode_reward += reward
            steps += 1
            
            # Lernen, wenn genügend Erfahrungen gesammelt wurden
            if steps % config.get('update_interval', 20) == 0:
                agent.learn()
        
        # Episode abgeschlossen
        all_rewards.append(episode_reward)
        all_makespans.append(info.get('makespan', 0))
    
    # Trainingsergebnisse
    results = {
        'final_avg_reward': np.mean(all_rewards[-20:]),
        'final_avg_makespan': np.mean(all_makespans[-20:]),
        'min_makespan': min(all_makespans)
    }
    
    return results

def evaluate_with_config(env, agent, n_episodes):
    """Evaluiert einen Agenten"""
    makespans = []
    rewards = []
    
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
    
    # Ergebnisse
    results = {
        'avg_makespan': np.mean(makespans),
        'min_makespan': np.min(makespans),
        'max_makespan': np.max(makespans),
        'std_makespan': np.std(makespans),
        'avg_reward': np.mean(rewards)
    }
    
    return results

def save_tuning_results(results, best_config):
    """Speichert die Ergebnisse der Hyperparameter-Optimierung"""
    results_dir = os.path.join(os.getcwd(), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Ergebnisse in JSON-Format konvertieren
    results_data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'best_config': best_config,
        'configurations': []
    }
    
    for result in results:
        results_data['configurations'].append({
            'config': result['config'],
            'train_results': result['train_results'],
            'eval_results': result['eval_results']
        })
    
    # In Datei speichern
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(results_dir, f"hyperparameter_tuning_{timestamp}.json")
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=4)
    
    # Beste Konfiguration separat speichern
    best_config_path = os.path.join(results_dir, f"best_config_{timestamp}.json")
    with open(best_config_path, 'w', encoding='utf-8') as f:
        json.dump(best_config, f, ensure_ascii=False, indent=4)
    
    print(f"\nErgebnisse gespeichert in: {filename}")
    print(f"Beste Konfiguration gespeichert in: {best_config_path}")

def visualize_tuning_results(results, param_names):
    """Visualisiert die Ergebnisse der Hyperparameter-Optimierung"""
    plt.figure(figsize=(15, 10))
    
    # Sortiere Ergebnisse nach Makespan
    sorted_results = sorted(results, key=lambda x: x['eval_results']['avg_makespan'])
    
    # Top-N Konfigurationen für die Visualisierung
    top_n = min(10, len(sorted_results))
    top_configs = sorted_results[:top_n]
    
    # Makespan-Vergleich
    plt.subplot(2, 1, 1)
    config_labels = [f"Config {i+1}" for i in range(top_n)]
    makespans = [config['eval_results']['avg_makespan'] for config in top_configs]
    
    bars = plt.bar(config_labels, makespans)
    plt.title('Top Konfigurationen nach Makespan')
    plt.ylabel('Durchschnittlicher Makespan')
    plt.grid(True, axis='y')
    
    # Werte über den Balken anzeigen
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                 f'{height:.0f}', ha='center', va='bottom')
    
    # Parameter-Werte für die besten Konfigurationen
    plt.subplot(2, 1, 2)
    
    # Für jeden Parameter einen eigenen Plot
    for i, param in enumerate(param_names):
        plt.subplot(2, len(param_names), len(param_names) + i + 1)
        
        # Parameter-Werte und zugehörige Makespan-Werte sammeln
        param_values = []
        param_makespans = []
        
        for result in results:
            param_values.append(result['config'][param])
            param_makespans.append(result['eval_results']['avg_makespan'])
        
        # Wenn der Parameter numerisch ist, Streudiagramm erstellen
        if isinstance(param_values[0], (int, float)):
            plt.scatter(param_values, param_makespans, alpha=0.7)
            plt.title(f'Einfluss von {param}')
            plt.xlabel(param)
            plt.ylabel('Makespan')
            plt.grid(True)
        # Sonst Boxplot für kategorische Parameter
        else:
            # Gruppiere Makespan-Werte nach Parameter-Werten
            param_groups = {}
            for val, makespan in zip(param_values, param_makespans):
                if val not in param_groups:
                    param_groups[val] = []
                param_groups[val].append(makespan)
            
            labels = list(param_groups.keys())
            data = [param_groups[label] for label in labels]
            
            plt.boxplot(data, labels=labels)
            plt.title(f'Einfluss von {param}')
            plt.ylabel('Makespan')
            plt.grid(True, axis='y')
    
    plt.tight_layout()
    
    # Speichern der Grafik
    plots_dir = os.path.join(os.getcwd(), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(plots_dir, f"hyperparameter_tuning_{timestamp}.png"))
    plt.show()

def main():
    """Hauptfunktion zum Ausführen der Hyperparameter-Optimierung"""
    import argparse
    from pathlib import Path
    import sys
    
    # Füge das Hauptverzeichnis zum Pfad hinzu, um Importe zu ermöglichen
    sys.path.append(str(Path(__file__).parent.parent))
    
    # Importiere Funktionen aus main.py
    from main import config, rl_config, load_data, generate_synthetic_data
    
    parser = argparse.ArgumentParser(description='Hyperparameter-Optimierung für Job-Scheduling')
    parser.add_argument('--data', type=str, default=None,
                        help='Pfad zur Datendatei (optional)')
    parser.add_argument('--episodes', type=int, default=300,
                        help='Anzahl der Trainingsepisoden pro Konfiguration')
    parser.add_argument('--eval-episodes', type=int, default=5,
                        help='Anzahl der Evaluierungsepisoden')
    args = parser.parse_args()
    
    # Daten laden oder generieren
    if args.data:
        data = load_data(args.data)
    else:
        data = generate_synthetic_data(config)
    
    # Parameter-Grid für die Suche definieren
    param_grid = {
        'lr': [0.0001, 0.0003, 0.0005],
        'entropy_coef': [0.005, 0.01, 0.02],
        'batch_size': [32, 64, 128],
        'n_epochs': [5, 10, 15]
    }
    
    # Grid-Search durchführen
    results, best_config = grid_search(
        data, 
        rl_config, 
        param_grid, 
        n_episodes=args.episodes, 
        n_eval_episodes=args.eval_episodes
    )
    
    print("\n=== Hyperparameter-Optimierung abgeschlossen ===")
    print("Beste Konfiguration:")
    for param, value in best_config.items():
        print(f"  {param}: {value}")

if __name__ == "__main__":
    main()