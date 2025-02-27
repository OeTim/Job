import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime
import sys
from pathlib import Path

# Füge das Hauptverzeichnis zum Pfad hinzu
sys.path.append(str(Path(__file__).parent.parent))

from environments.job_env import JobSchedulingEnv
from agents.ppo_agent import PPOAgent
from main import config, rl_config, load_data, generate_synthetic_data

def train_with_tracking(env, agent, config, track_interval=10):
    """
    Trainiert den Agenten und trackt detaillierte Metriken für Lernkurven
    
    Args:
        env: Die Umgebung
        agent: Der PPO-Agent
        config: Konfigurationsparameter
        track_interval: Intervall für das Tracking von Metriken
    
    Returns:
        dict: Tracking-Daten
    """
    # Trainingskonfiguration
    n_episodes = config.get('n_episodes', 1000)
    max_steps = config.get('max_steps', 100)
    update_interval = config.get('update_interval', 20)
    
    # Tracking-Variablen
    tracking_data = {
        'episodes': [],
        'rewards': [],
        'makespans': [],
        'avg_rewards': [],
        'avg_makespans': [],
        'baseline_comparisons': []
    }
    
    # Baseline-Strategien für Vergleich
    strategies = ['FIFO', 'LIFO', 'SPT']
    
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
        
        # Makespan erfassen
        makespan = info.get('makespan', 0)
        
        # Tracking in bestimmten Intervallen
        if episode % track_interval == 0:
            tracking_data['episodes'].append(episode)
            tracking_data['rewards'].append(episode_reward)
            tracking_data['makespans'].append(makespan)
            
            # Gleitende Durchschnitte
            window = min(20, len(tracking_data['rewards']))
            tracking_data['avg_rewards'].append(np.mean(tracking_data['rewards'][-window:]))
            tracking_data['avg_makespans'].append(np.mean(tracking_data['makespans'][-window:]))
            
            #