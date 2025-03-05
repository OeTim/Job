import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import networkx as nx
from graph_transformer import GraphTransformer, extract_graph_features

class PolicyNetwork(nn.Module):
    """
    Policy-Netzwerk für den PPO-Agenten.
    Kombiniert Graph Transformer mit MLP für die Aktionsauswahl.
    """
    def __init__(self, graph_transformer: GraphTransformer, state_dim: int, action_dim: int):
        super(PolicyNetwork, self).__init__()
        self.graph_transformer = graph_transformer
        
        # Feature-Dimension nach Graph Transformer
        graph_feature_dim = graph_transformer.output_dim
        
        # MLP für die Kombination von Graph-Features und Zustandsfeatures
        self.fc1 = nn.Linear(graph_feature_dim + state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)
        
    def forward(self, state_dict, graph_features, adj_matrix):
        # Verarbeite den Graphen mit dem Transformer
        graph_embedding = self.graph_transformer(graph_features, adj_matrix)
        
        # Aggregiere die Graph-Embedding (Mittelwert über alle Knoten)
        graph_embedding = torch.mean(graph_embedding, dim=0, keepdim=True)
        
        # Konvertiere den Zustandsdict in einen Tensor
        state_tensor = torch.FloatTensor([
            state_dict['remaining_ops'],
            state_dict['avg_processing_time'][0],
            *state_dict['machine_utilization'],
            state_dict['critical_path_length'][0],
            state_dict['machine_conflicts'],
            state_dict['current_time'][0]
        ]).unsqueeze(0)
        
        # Kombiniere Graph-Embedding und Zustandsfeatures
        x = torch.cat([graph_embedding, state_tensor], dim=1)
        
        # MLP
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Ausgabe: Logits für jede Aktion
        action_logits = self.fc3(x)
        
        return action_logits
    
    def get_action_probs(self, state_dict, graph_features, adj_matrix):
        logits = self.forward(state_dict, graph_features, adj_matrix)
        return F.softmax(logits, dim=1)
    
    def get_action(self, state_dict, graph_features, adj_matrix):
        with torch.no_grad():
            probs = self.get_action_probs(state_dict, graph_features, adj_matrix)
            action = torch.multinomial(probs, 1).item()
        return action

class ValueNetwork(nn.Module):
    """
    Value-Netzwerk für den PPO-Agenten.
    Schätzt den erwarteten Reward für einen Zustand.
    """
    def __init__(self, graph_transformer: GraphTransformer, state_dim: int):
        super(ValueNetwork, self).__init__()
        self.graph_transformer = graph_transformer
        
        # Feature-Dimension nach Graph Transformer
        graph_feature_dim = graph_transformer.output_dim
        
        # MLP für die Kombination von Graph-Features und Zustandsfeatures
        self.fc1 = nn.Linear(graph_feature_dim + state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, state_dict, graph_features, adj_matrix):
        # Verarbeite den Graphen mit dem Transformer
        graph_embedding = self.graph_transformer(graph_features, adj_matrix)
        
        # Aggregiere die Graph-Embedding (Mittelwert über alle Knoten)
        graph_embedding = torch.mean(graph_embedding, dim=0, keepdim=True)
        
        # Konvertiere den Zustandsdict in einen Tensor
        state_tensor = torch.FloatTensor([
            state_dict['remaining_ops'],
            state_dict['avg_processing_time'][0],
            *state_dict['machine_utilization'],
            state_dict['critical_path_length'][0],
            state_dict['machine_conflicts'],
            state_dict['current_time'][0]
        ]).unsqueeze(0)
        
        # Kombiniere Graph-Embedding und Zustandsfeatures
        x = torch.cat([graph_embedding, state_tensor], dim=1)
        
        # MLP
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Ausgabe: Geschätzter Wert des Zustands
        value = self.fc3(x)
        
        return value

class PPOAgent:
    """
    Proximal Policy Optimization (PPO) Agent für Job-Shop Scheduling.
    """
    def __init__(self, env, graph: nx.DiGraph, learning_rate=0.0003, gamma=0.99, 
                 clip_epsilon=0.2, epochs=10, batch_size=64):
        self.env = env
        self.graph = graph
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Extrahiere Graph-Features und Adjazenzmatrix
        self.graph_features, self.adj_matrix = extract_graph_features(graph)
        
        # Bestimme die Dimensionen
        state_dim = 1 + 1 + 10 + 1 + 1 + 1  # remaining_ops, avg_time, machine_util, crit_path, conflicts, time
        action_dim = env.action_space.n
        
        # Erstelle Graph Transformer
        self.graph_transformer = GraphTransformer(
            input_dim=self.graph_features.shape[1],
            hidden_dim=64,
            output_dim=32,
            num_heads=4,
            num_layers=2,
            dropout=0.1
        )
        
        # Erstelle Policy- und Value-Netzwerke
        self.policy = PolicyNetwork(self.graph_transformer, state_dim, action_dim)
        self.value = ValueNetwork(self.graph_transformer, state_dim)
        
        # Optimierer
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=learning_rate)
        
    def select_action(self, state):
        """Wählt eine Aktion basierend auf dem aktuellen Zustand."""
        return self.policy.get_action(state, self.graph_features, self.adj_matrix)
    
    def train(self, num_episodes=1000):
        """Trainiert den Agenten für die angegebene Anzahl von Episoden."""
        best_reward = float('-inf')
        
        for episode in range(num_episodes):
            # Sammle Trainingsdaten
            states, actions, rewards, log_probs, values = [], [], [], [], []
            state = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Wähle Aktion
                action = self.select_action(state)
                action_probs = self.policy.get_action_probs(state, self.graph_features, self.adj_matrix)
                value = self.value(state, self.graph_features, self.adj_matrix)
                
                # Führe Aktion aus
                next_state, reward, done, _ = self.env.step(action)
                
                # Speichere Transition
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(torch.log(action_probs[0, action]))
                values.append(value)
                
                state = next_state
                episode_reward += reward
            
            # Berechne Vorteile und Returns
            returns = self._compute_returns(rewards)
            advantages = self._compute_advantages(returns, values)
            
            # Aktualisiere Policy und Value Netzwerk
            self._update_policy(states, actions, log_probs, advantages)
            self._update_value(states, returns)
            
            # Logging
            if episode % 10 == 0:
                print(f"Episode {episode}: Reward = {episode_reward:.2f}")
            
            # Speichere bestes Modell
            if episode_reward > best_reward:
                best_reward = episode_reward
                self._save_model()
    
    def _compute_returns(self, rewards: List[float]) -> torch.Tensor:
        """Berechnet die diskontierten Returns."""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns)
    
    def _compute_advantages(self, returns: torch.Tensor, values: List[torch.Tensor]) -> torch.Tensor:
        """Berechnet die Advantage-Werte."""
        values = torch.cat(values).squeeze()
        advantages = returns - values.detach()
        return advantages
    
    def _update_policy(self, states, actions, old_log_probs, advantages):
        """Aktualisiert das Policy-Netzwerk mit PPO."""
        for _ in range(self.epochs):
            for batch_idx in range(0, len(states), self.batch_size):
                batch_states = states[batch_idx:batch_idx + self.batch_size]
                batch_actions = actions[batch_idx:batch_idx + self.batch_size]
                batch_old_log_probs = old_log_probs[batch_idx:batch_idx + self.batch_size]
                batch_advantages = advantages[batch_idx:batch_idx + self.batch_size]
                
                # Berechne neue Action Probabilities
                action_probs = self.policy.get_action_probs(batch_states[0], self.graph_features, self.adj_matrix)
                new_log_probs = torch.log(action_probs[0, batch_actions[0]])
                
                # Berechne Ratio und geclippten Verlust
                ratio = torch.exp(new_log_probs - batch_old_log_probs[0])
                surr1 = ratio * batch_advantages[0]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages[0]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Optimiere Policy
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()
    
    def _update_value(self, states, returns):
        """Aktualisiert das Value-Netzwerk."""
        for _ in range(self.epochs):
            for batch_idx in range(0, len(states), self.batch_size):
                batch_states = states[batch_idx:batch_idx + self.batch_size]
                batch_returns = returns[batch_idx:batch_idx + self.batch_size]
                
                # Berechne Value Loss
                values = self.value(batch_states[0], self.graph_features, self.adj_matrix)
                value_loss = F.mse_loss(values, batch_returns.unsqueeze(1))
                
                # Optimiere Value
                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()
    
    def _save_model(self):
        """Speichert das aktuelle Modell."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'graph_transformer_state_dict': self.graph_transformer.state_dict()
        }, 'best_model.pth')