import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import torch.nn.functional as F
import random

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward):
        super(TransformerEncoder, self).__init__()
        
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x, mask=None):
        x = self.embedding(x)
        return self.transformer_encoder(x, src_key_padding_mask=mask)

class PPONetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, config):
        super(PPONetwork, self).__init__()
        
        # Konfigurationsparameter
        d_model = config.get('d_model', 128)
        nhead = config.get('nhead', 4)
        num_layers = config.get('num_layers', 2)
        dim_feedforward = config.get('dim_feedforward', 512)
        
        # Transformer für Job-Features
        self.job_transformer = TransformerEncoder(
            input_dim=10,  # Job-Feature-Dimension
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward
        )
        
        # Transformer für Maschinen-Features
        self.machine_transformer = TransformerEncoder(
            input_dim=3,  # Maschinen-Feature-Dimension
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward
        )
        
        # Gemeinsame Feature-Verarbeitung
        self.fc1 = nn.Linear(d_model * 2 + 3, 256)  # +3 für Zeit-Features
        self.fc2 = nn.Linear(256, 128)
        
        # Actor (Policy) und Critic (Value) Heads
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)
        
    def forward(self, obs):
        # Extrahiere Features aus dem Beobachtungsraum
        job_features = obs['waiting_jobs']
        machine_features = obs['machine_status']
        time_features = obs['time_features']
        
        # Erstelle Masken für Padding (falls nötig)
        job_mask = torch.all(job_features == 0, dim=2) if len(job_features.shape) > 2 else None
        machine_mask = torch.all(machine_features == 0, dim=2) if len(machine_features.shape) > 2 else None
        
        # Verarbeite Features mit Transformern
        job_encoded = self.job_transformer(job_features, job_mask)
        machine_encoded = self.machine_transformer(machine_features, machine_mask)
        
        # Global Pooling über die Sequenzdimension
        job_pooled = torch.mean(job_encoded, dim=1)
        machine_pooled = torch.mean(machine_encoded, dim=1)
        
        # Konkateniere alle Features
        combined = torch.cat([job_pooled, machine_pooled, time_features], dim=1)
        
        # Gemeinsame Feature-Verarbeitung
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        
        # Actor und Critic Outputs
        action_probs = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        
        return action_probs, value

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
        
    def store(self, state, action, prob, val, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)
        
    def clear(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        
    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return batches

class PPOAgent:
    # In der PPOAgent-Klasse:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        
        # Hyperparameter
        self.gamma = config.get('gamma', 0.99)
        self.policy_clip = config.get('policy_clip', 0.2)
        self.n_epochs = config.get('n_epochs', 10)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.batch_size = config.get('batch_size', 64)
        self.lr = config.get('lr', 0.0003)  # Lernrate hinzugefügt
        self.entropy_coef = config.get('entropy_coef', 0.01)  # Entropie-Koeffizient hinzugefügt
        
        # Add epsilon decay for better exploration
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay = 0.997  # Langsamerer Decay
        self.current_epsilon = self.epsilon_start
        
        # Add learning rate decay
        self.lr_decay = 0.999
        self.min_lr = 0.00001
        
        # Netzwerk und Optimierer
        obs_dim = {
            'waiting_jobs': env.observation_space['waiting_jobs'].shape,
            'machine_status': env.observation_space['machine_status'].shape,
            'time_features': env.observation_space['time_features'].shape
        }
        action_dim = env.action_space.n
        
        self.actor_critic = PPONetwork(obs_dim, action_dim, config)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.lr)
        
        # Speicher für Erfahrungen
        self.memory = PPOMemory(self.batch_size)
        
        # Gerät (CPU/GPU)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.actor_critic.to(self.device)
        
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store(state, action, probs, vals, reward, done)
        
    def choose_action(self, observation, eval_mode=False):
        # Konvertiere Beobachtung zu Tensor
        state = {}
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                state[key] = torch.tensor(value, dtype=torch.float32).unsqueeze(0).to(self.device)
            else:
                state[key] = torch.tensor([value], dtype=torch.float32).to(self.device)
        
        # Forward pass durch das Netzwerk
        action_probs, value = self.actor_critic(state)
        value = torch.squeeze(value).item()
        
        if eval_mode:
            # Im Evaluationsmodus: Wähle die Aktion mit der höchsten Wahrscheinlichkeit
            action = torch.argmax(action_probs).item()
            log_prob = 0  # Nicht relevant im Evaluationsmodus
        else:
            # Im Trainingsmodus: Exploration mit Epsilon-Greedy
            if random.random() < self.current_epsilon:
                # Completely random action
                action = random.randint(0, self.env.action_space.n - 1)
                # Create categorical distribution from action_probs
                dist = Categorical(action_probs)
                log_prob = dist.log_prob(torch.tensor(action, device=self.device)).item()
            else:
                # Use policy with temperature for controlled exploration
                temperature = 1.0 + self.entropy_coef * 5  # Reduced multiplier
                # Apply temperature to action_probs before creating Categorical
                scaled_probs = action_probs / temperature
                dist = Categorical(scaled_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action).item()
                action = action.item()
            
            # Decay epsilon
            self.current_epsilon = max(self.epsilon_end, 
                                 self.current_epsilon * self.epsilon_decay)
        
        return action, log_prob, value
    
    # In der learn-Methode:
    def learn(self):
        for _ in range(self.n_epochs):
            # Generiere Batches
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, done_arr = self._prepare_tensors()
            batches = self.memory.generate_batches()
            
            # Berechne Vorteile
            advantages = self._compute_advantages(vals_arr, reward_arr, done_arr)
            
            # Iteriere über Batches
            for batch in batches:
                # Extrahiere Batch-Daten
                states_batch = {k: v[batch] for k, v in state_arr.items()}
                actions_batch = action_arr[batch]
                old_probs_batch = old_prob_arr[batch]
                advantages_batch = advantages[batch]
                
                # Forward pass
                dist, critic_value = self.actor_critic(states_batch)
                
                # Berechne Critic Loss
                critic_value = torch.squeeze(critic_value)
                returns = advantages_batch + vals_arr[batch]
                critic_loss = F.mse_loss(critic_value, returns)
                
                # Berechne Actor Loss
                dist = Categorical(dist)
                new_probs = dist.log_prob(actions_batch)
                prob_ratio = torch.exp(new_probs - old_probs_batch)
                weighted_probs = advantages_batch * prob_ratio
                clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * advantages_batch
                actor_loss = -torch.min(weighted_probs, clipped_probs).mean()
                
                # Entropie-Bonus für Exploration
                entropy = dist.entropy().mean()
                
                # Gesamtverlust
                total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy
                
                # Optimierungsschritt
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizer.step()
        
        # Speicher leeren
        self.memory.clear()
    
    def _prepare_tensors(self):
        # Konvertiere gespeicherte Daten zu Tensoren
        states = self.memory.states
        actions = torch.tensor(self.memory.actions, dtype=torch.int64).to(self.device)
        old_probs = torch.tensor(self.memory.probs, dtype=torch.float32).to(self.device)
        vals = torch.tensor(self.memory.vals, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(self.memory.rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(self.memory.dones, dtype=torch.bool).to(self.device)
        
        # Konvertiere states zu Dictionary von Tensoren
        state_dict = {
            'waiting_jobs': torch.zeros((len(states), *states[0]['waiting_jobs'].shape), dtype=torch.float32).to(self.device),
            'machine_status': torch.zeros((len(states), *states[0]['machine_status'].shape), dtype=torch.float32).to(self.device),
            'time_features': torch.zeros((len(states), *states[0]['time_features'].shape), dtype=torch.float32).to(self.device)
        }
        
        for i, state in enumerate(states):
            for key in state_dict.keys():
                state_dict[key][i] = torch.tensor(state[key], dtype=torch.float32)
        
        return state_dict, actions, old_probs, vals, rewards, dones
    
    def _compute_advantages(self, values, rewards, dones):
        advantages = torch.zeros_like(rewards).to(self.device)
        last_advantage = 0
        last_value = 0
        
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t].float()
            last_value = last_value * mask
            last_advantage = last_advantage * mask
            
            delta = rewards[t] + self.gamma * last_value - values[t]
            last_advantage = delta + self.gamma * self.gae_lambda * last_advantage
            advantages[t] = last_advantage
            last_value = values[t]
            
        return advantages
    
    def save_models(self, path):
        torch.save(self.actor_critic.state_dict(), path)
        
    def load_models(self, path):
        self.actor_critic.load_state_dict(torch.load(path))

class SimplerPPONetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, config):
        super(SimplerPPONetwork, self).__init__()
        
        # Einfachere Feature-Verarbeitung
        self.job_encoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        self.machine_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # Gemeinsame Verarbeitung
        self.shared = nn.Sequential(
            nn.Linear(48 + 3, 128),  # 32 + 16 + 3 (time features)
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)