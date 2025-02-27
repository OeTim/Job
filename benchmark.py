import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import argparse
from datetime import datetime
from pathlib import Path

from environments.job_env import JobSchedulingEnv
from agents.ppo_agent import PPOAgent
from evaluate import evaluate_agent, compare_strategies, visualize_schedule

def load_model(model_path, env, config):
    """
    Lädt ein trainiertes Modell und passt die Umgebung an die Modellgröße an
    
    Args:
        model_path: Pfad zum Modell
        env: Ursprüngliche Umgebung
        config: Konfiguration
        
    Returns:
        agent: Der geladene Agent
        env: Die angepasste Umgebung
    """
    # Modellname extrahieren
    model_name = Path(model_path).stem
    print(f"Lade Modell: {model_name}")
    
    # Modell-Parameter laden, um die Dimensionen zu prüfen
    model_state = torch.load(model_path)
    
    # Anzahl der Aktionen aus dem Modell bestimmen
    if 'actor.weight' in model_state:
        model_action_dim = model_state['actor.weight'].shape[0]
        print(f"Modell wurde für {model_action_dim} Aktionen trainiert")
        
        # Prüfen, ob die Umgebung die gleiche Anzahl von Aktionen hat
        current_action_dim = env.action_space.n
        if model_action_dim != current_action_dim:
            print(f"Warnung: Modell erwartet {model_action_dim} Aktionen, aber Umgebung hat {current_action_dim}")
            print("Erstelle neue Umgebung mit passender Jobanzahl...")
            
            # Neue Daten mit der richtigen Anzahl von Jobs generieren
            from main import generate_synthetic_data
            
            # Konfiguration anpassen
            custom_config = config.copy()
            custom_config['n_jobs'] = model_action_dim
            
            # Neue Daten generieren
            new_data = generate_synthetic_data(custom_config)
            
            # Neue Umgebung erstellen
            env = JobSchedulingEnv(new_data, config)
            print(f"Neue Umgebung mit {model_action_dim} Jobs erstellt")
    
    # Agent erstellen und Modell laden
    agent = PPOAgent(env, config)
    agent.load_models(model_path)
    print(f"Modell geladen aus: {model_path}")
    
    return agent, env

def benchmark_models(models_list, data, config, rl_config, n_episodes=10, save_results=True):
    """
    Vergleicht mehrere Modelle auf dem gleichen Datensatz
    
    Args:
        models_list: Liste von Modellpfaden
        data: Produktionsdaten
        config: Umgebungskonfiguration
        rl_config: RL-Konfiguration
        n_episodes: Anzahl der Evaluierungsepisoden pro Modell
        save_results: Ob Ergebnisse gespeichert werden sollen
    """
    results = {}
    env = JobSchedulingEnv(data, config)
    
    # Baseline-Strategien evaluieren
    baseline_results = {}
    strategies = ['FIFO', 'LIFO', 'SPT']
    
    for strategy in strategies:
        env.reset()
        _, _, _, info = env.step(strategy)
        baseline_results[strategy] = info.get('makespan', 0)
    
    # Modelle evaluieren
    for model_path in models_list:
        try:
            model_name = Path(model_path).stem
            agent, model_env = load_model(model_path, env, rl_config)
            
            print(f"\nEvaluiere Modell: {model_name}")
            model_results = evaluate_agent(model_env, agent, n_episodes)
            
            # Prüfen, ob die Makespans variieren
            if model_results['std_makespan'] < 0.01:
                print("\nHinweis: Die Makespan-Werte variieren nicht. Dies kann normal sein, wenn:")
                print("  1. Der Agent eine deterministische Policy gelernt hat")
                print("  2. Die Umgebung keine stochastischen Elemente enthält")
                print("  3. Die Anzahl der Evaluierungsepisoden zu gering ist")
                print("\nUm mehr Variation zu sehen, können Sie:")
                print("  - Die Umgebung mit mehr Zufallselementen konfigurieren")
                print("  - Die Temperatur bei der Aktionsauswahl während der Evaluation erhöhen")
                print("  - Verschiedene Datensätze für die Evaluation verwenden")
            
            results[model_name] = {
                'avg_makespan': model_results['avg_makespan'],
                'min_makespan': model_results['min_makespan'],
                'max_makespan': model_results['max_makespan'],
                'std_makespan': model_results['std_makespan'],
                'avg_reward': model_results['avg_reward'],
                'best_schedule': model_results['best_schedule']
            }
            
            # Besten Schedule visualisieren
            print(f"\nBester Schedule für {model_name}:")
            visualize_schedule(model_results['best_schedule'], f"Bester Schedule - {model_name}")
        
        except Exception as e:
            print(f"\nFehler beim Evaluieren von {model_path}: {e}")
            print("Überspringe dieses Modell und fahre mit dem nächsten fort.")
            continue
    
    # Rest der Funktion bleibt unverändert...
    results[model_name] = {
        'avg_makespan': model_results['avg_makespan'],
        'min_makespan': model_results['min_makespan'],
        'max_makespan': model_results['max_makespan'],
        'std_makespan': model_results['std_makespan'],
        'avg_reward': model_results['avg_reward'],
        'best_schedule': model_results['best_schedule']
    }
    
    # Besten Schedule visualisieren
    print(f"\nBester Schedule für {model_name}:")
    visualize_schedule(model_results['best_schedule'], f"Bester Schedule - {model_name}")

    # Ergebnisse zusammenfassen
    print("\n=== Benchmark-Ergebnisse ===")
    print("\nBaseline-Strategien:")
    for strategy, makespan in baseline_results.items():
        print(f"  {strategy}: Makespan = {makespan:.2f}")
    
    print("\nTrainierte Modelle:")
    for model_name, model_results in results.items():
        print(f"  {model_name}:")
        print(f"    Durchschnittlicher Makespan: {model_results['avg_makespan']:.2f}")
        print(f"    Bester Makespan: {model_results['min_makespan']:.2f}")
        print(f"    Standardabweichung: {model_results['std_makespan']:.2f}")
    
    # Visualisierung der Ergebnisse
    visualize_benchmark_results(baseline_results, results)
    
    # Ergebnisse speichern
    if save_results:
        save_benchmark_results(baseline_results, results)
    
    return baseline_results, results

def visualize_benchmark_results(baseline_results, model_results):
    """Visualisiert die Benchmark-Ergebnisse"""
    # Makespan-Vergleich
    plt.figure(figsize=(12, 6))
    
    # Baseline-Strategien
    baseline_names = list(baseline_results.keys())
    baseline_values = [baseline_results[name] for name in baseline_names]
    
    # Modell-Ergebnisse
    model_names = list(model_results.keys())
    model_avg_values = [model_results[name]['avg_makespan'] for name in model_names]
    model_min_values = [model_results[name]['min_makespan'] for name in model_names]
    
    # Alle Namen und Werte kombinieren
    all_names = baseline_names + [f"{name} (Avg)" for name in model_names] + [f"{name} (Best)" for name in model_names]
    all_values = baseline_values + model_avg_values + model_min_values
    
    # Farben für die Balken
    colors = ['lightgray'] * len(baseline_names) + ['skyblue'] * len(model_names) + ['green'] * len(model_names)
    
    # Balkendiagramm erstellen
    bars = plt.bar(range(len(all_names)), all_values, color=colors)
    plt.xticks(range(len(all_names)), all_names, rotation=45, ha='right')
    plt.title('Vergleich der Makespan-Werte')
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
    plt.savefig(os.path.join(plots_dir, f"benchmark_comparison_{timestamp}.png"))
    plt.show()

def save_benchmark_results(baseline_results, model_results):
    """Speichert die Benchmark-Ergebnisse"""
    results_dir = os.path.join(os.getcwd(), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Ergebnisse in JSON-Format konvertieren
    results_data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'baseline_strategies': baseline_results,
        'models': {}
    }
    
    for model_name, model_data in model_results.items():
        # Schedules können nicht direkt in JSON gespeichert werden
        model_data_copy = model_data.copy()
        if 'best_schedule' in model_data_copy:
            # Vereinfachte Version des Schedules speichern
            model_data_copy['best_schedule'] = [
                {k: v for k, v in op.items() if k != 'job_object'} 
                for op in model_data_copy['best_schedule']
            ]
        results_data['models'][model_name] = model_data_copy
    
    # In Datei speichern
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(results_dir, f"benchmark_results_{timestamp}.json")
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=4)
    
    print(f"\nBenchmark-Ergebnisse gespeichert in: {filename}")

def main():
    parser = argparse.ArgumentParser(description='Benchmark für Job-Scheduling-Modelle')
    parser.add_argument('--models', nargs='+', required=True,
                        help='Pfade zu den zu vergleichenden Modellen')
    parser.add_argument('--data', type=str, default=None,
                        help='Pfad zur Datendatei (optional)')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Anzahl der Evaluierungsepisoden pro Modell')
    args = parser.parse_args()
    
    # Konfigurationen laden
    from main import config, rl_config, load_data, generate_synthetic_data
    
    # Daten laden oder generieren
    if args.data:
        data = load_data(args.data)
    else:
        data = generate_synthetic_data(config)
    
    # Benchmark durchführen
    benchmark_models(args.models, data, config, rl_config, n_episodes=args.episodes)

if __name__ == "__main__":
    main()