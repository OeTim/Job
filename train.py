import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

from environments.job_env import JobSchedulingEnv
from agents.ppo_agent import PPOAgent

def train_ppo(env, agent, config):
    """
    Trainiert den PPO-Agenten auf der Job-Scheduling-Umgebung
    
    Args:
        env: Die Umgebung
        agent: Der PPO-Agent
        config: Konfigurationsparameter
    
    Returns:
        dict: Trainingsergebnisse
    """
    # Trainingskonfiguration
    n_episodes = config.get('n_episodes', 1000)
    max_steps = config.get('max_steps', 100)
    update_interval = config.get('update_interval', 20)
    eval_interval = config.get('eval_interval', 50)
    
    # Tracking-Variablen
    all_rewards = []
    episode_rewards = []
    all_makespans = []
    best_makespan = float('inf')
    best_model_path = None
    
    # Verzeichnis für Modelle erstellen
    models_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Trainingsschleife
    for episode in tqdm(range(1, n_episodes + 1), desc="Training"):
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
        # In der train_ppo Funktion, beim Speichern des Modells:
        if makespan < best_makespan and makespan > 0:
            best_makespan = makespan
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Füge Metadaten zum Modellnamen hinzu
            n_jobs = len(env.jobs)
            best_model_path = os.path.join(models_dir, f"best_model_{timestamp}_jobs_{n_jobs}_makespan_{int(best_makespan)}.pt")
            
            # Speichere auch die Konfiguration und Umgebungsparameter
            model_info = {
                'timestamp': timestamp,
                'makespan': best_makespan,
                'n_jobs': n_jobs,
                'config': {k: str(v) if isinstance(v, (type, torch.device)) else v for k, v in config.items()}
            }
            
            info_path = os.path.join(models_dir, f"model_info_{timestamp}.json")
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, ensure_ascii=False, indent=4)
            
            agent.save_models(best_model_path)
            print(f"\nNeuer bester Makespan: {best_makespan:.2f} - Modell gespeichert in {best_model_path}")
        
        # Regelmäßige Evaluation
        if episode % eval_interval == 0:
            avg_reward = np.mean(episode_rewards[-eval_interval:])
            avg_makespan = np.mean(all_makespans[-eval_interval:])
            print(f"\nEpisode {episode}: Durchschnittliche Belohnung = {avg_reward:.2f}, Durchschnittlicher Makespan = {avg_makespan:.2f}")
            
            # Vergleich mit Baseline-Strategien
            compare_with_baselines(env)
    
    # Trainingsergebnisse
    results = {
        'rewards': all_rewards,
        'makespans': all_makespans,
        'best_makespan': best_makespan,
        'best_model_path': best_model_path
    }
    
    # Ergebnisse visualisieren
    plot_training_results(results)
    
    return results

def compare_with_baselines(env):
    """Vergleicht den aktuellen Agenten mit Baseline-Strategien"""
    strategies = ['FIFO', 'LIFO', 'SPT']
    results = {}
    
    for strategy in strategies:
        env_copy = env  # In einer realen Implementierung sollte hier eine Kopie erstellt werden
        env_copy.reset()
        _, _, _, info = env_copy.step(strategy)
        results[strategy] = info.get('makespan', 0)
    
    print("\nVergleich mit Baseline-Strategien:")
    for strategy, makespan in results.items():
        print(f"  {strategy}: Makespan = {makespan:.2f}")
    
    return results

def plot_training_results(results):
    """Visualisiert die Trainingsergebnisse"""
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
    plots_dir = os.path.join(os.getcwd(), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(plots_dir, f"training_results_{timestamp}.png"))
    plt.show()

def save_config(config, results=None):
    """Speichert die Konfiguration und Ergebnisse"""
    config_dir = os.path.join(os.getcwd(), 'configs')
    os.makedirs(config_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = os.path.join(config_dir, f"config_{timestamp}.json")
    
    # Ergebnisse hinzufügen, falls vorhanden
    if results:
        # Nur serialisierbare Daten speichern
        serializable_results = {
            'best_makespan': results.get('best_makespan', 0),
            'best_model_path': results.get('best_model_path', ''),
            'final_reward': results.get('rewards', [])[-1] if results.get('rewards') else 0,
            'final_makespan': results.get('makespans', [])[-1] if results.get('makespans') else 0
        }
        config['results'] = serializable_results
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Konfiguration gespeichert in: {config_path}")
    return config_path