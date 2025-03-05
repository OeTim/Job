# Reward-Mechanismus und Makespan-Optimierung im Job-Scheduling-System

## Inhaltsverzeichnis
1. [Einführung](#einführung)
2. [Reward-Funktion im Detail](#reward-funktion-im-detail)
3. [Zusammenhang zwischen Reward und Makespan](#zusammenhang-zwischen-reward-und-makespan)
4. [Optimierungsmöglichkeiten](#optimierungsmöglichkeiten)
5. [Implementierungsdetails](#implementierungsdetails)

## Einführung

In unserem Job-Scheduling-System verwenden wir Reinforcement Learning (RL), um die optimale Reihenfolge von Operationen zu finden und damit den Makespan zu minimieren. Der Makespan ist die Gesamtzeit, die benötigt wird, um alle Jobs abzuschließen, also die Zeit vom Start bis zur Fertigstellung des letzten Jobs.

Die zentrale Komponente eines RL-Systems ist die Reward-Funktion, die dem Agenten Feedback darüber gibt, wie gut seine Aktionen sind. In unserem Fall ist die Reward-Funktion so gestaltet, dass sie den Agenten dazu anleitet, Entscheidungen zu treffen, die zu einem kürzeren Makespan führen.

## Reward-Funktion im Detail

Die Reward-Funktion ist in der `step`-Methode der `JobSchedulingEnv`-Klasse implementiert. Hier ist der relevante Code-Ausschnitt:

# Belohnung berechnen
prev_makespan = self.makespan
new_makespan = max(self.makespan, completion_time)
makespan_diff = new_makespan - prev_makespan

# Einfache Belohnung basierend auf Makespan-Änderung
reward = -makespan_diff

# Fortschrittsbonus
progress = len(self.completed_jobs) / len(self.jobs)
reward += progress * 10

# Zusätzlicher Bonus für Fertigstellung aller Jobs
if len(self.remaining_jobs) == 0:
    reward += 50

# Makespan aktualisieren
self.makespan = new_makespan



Lassen wir uns diese Zeilen im Detail anschauen:

1. prev_makespan = self.makespan : Speichert den aktuellen Makespan vor der Ausführung der Operation.
2. new_makespan = max(self.makespan, completion_time) : Berechnet den neuen Makespan nach Ausführung der Operation. Der Makespan ist das Maximum aus dem bisherigen Makespan und der Fertigstellungszeit der aktuellen Operation.
3. makespan_diff = new_makespan - prev_makespan : Berechnet die Differenz zwischen dem neuen und dem alten Makespan. Wenn die Operation den Makespan verlängert, ist dieser Wert positiv.
4. reward = -makespan_diff : Die grundlegende Belohnung ist negativ proportional zur Makespan-Erhöhung. Das bedeutet:
   
   - Wenn die Operation den Makespan nicht erhöht (makespan_diff = 0), ist die Belohnung 0.
   - Wenn die Operation den Makespan erhöht (makespan_diff > 0), ist die Belohnung negativ.
   Diese negative Belohnung motiviert den Agenten, Operationen zu wählen, die den Makespan möglichst wenig erhöhen.
5. progress = len(self.completed_jobs) / len(self.jobs) : Berechnet den Fortschritt als Verhältnis der abgeschlossenen Jobs zur Gesamtzahl der Jobs.
6. reward += progress * 10 : Fügt einen Bonus basierend auf dem Fortschritt hinzu. Je mehr Jobs abgeschlossen sind, desto höher ist dieser Bonus. Dies ermutigt den Agenten, Jobs abzuschließen.
7. if len(self.remaining_jobs) == 0: reward += 50 : Gibt einen großen Bonus, wenn alle Jobs abgeschlossen sind. Dies ist ein starker Anreiz, alle Jobs zu beenden.
8. self.makespan = new_makespan : Aktualisiert den Makespan für den nächsten Schritt.


## Zusammenhang zwischen Reward und Makespan
Der Zusammenhang zwischen dem Reward und dem Makespan ist invers: Je niedriger der Makespan, desto höher der Reward. Dies wird durch folgende Mechanismen erreicht:

1. Negative Belohnung für Makespan-Erhöhung : Der Agent erhält eine negative Belohnung proportional zur Erhöhung des Makespans. Dies motiviert ihn, Operationen zu wählen, die den Makespan minimal erhöhen oder idealerweise gar nicht.
2. Fortschrittsbonus : Der Agent erhält einen Bonus basierend auf dem Fortschritt. Dies verhindert, dass der Agent in einem lokalen Optimum stecken bleibt, bei dem er keine Operationen mehr ausführt, um negative Belohnungen zu vermeiden.
3. Abschlussbonus : Der große Bonus für den Abschluss aller Jobs stellt sicher, dass der Agent motiviert ist, alle Jobs zu beenden, auch wenn dies kurzfristig zu einer Erhöhung des Makespans führen könnte.
## Optimierungsmöglichkeiten
Die aktuelle Reward-Funktion ist relativ einfach und könnte auf verschiedene Weisen verbessert werden:

### 1. Normalisierung des Makespan-Unterschieds

        # Normalisierte Makespan-Änderung
        makespan_penalty = -makespan_diff / 100  # Normalisiere große Zahlen

Dies würde dazu beitragen, dass die Belohnung für die Makespan-Änderung in einem ähnlichen Bereich liegt wie die anderen Belohnungskomponenten.

### 2. Effizienzbonus für kurze Bearbeitungszeiten

        # Belohnung für effiziente Operationen
        efficiency_bonus = 0
        if processing_time > 0:
            efficiency = 1.0 / (processing_time + changeover_time)  # Höhere Belohnung für kürzere Bearbeitungszeiten
            efficiency_bonus = efficiency * 10

Dies würde den Agenten dazu ermutigen, Operationen mit kürzeren Bearbeitungszeiten zu bevorzugen, was oft zu einem kürzeren Makespan führt.

### 3. Strafe für Umrüstzeiten

        # Strafe für Umrüstzeiten
        changeover_penalty = -changeover_time / 10 if changeover_time > 0 else 0

Dies würde den Agenten dazu ermutigen, unnötige Umrüstungen zu vermeiden, was ebenfalls zu einem kürzeren Makespan beitragen kann.

### 4. Kombinierte verbesserte Reward-Funktion

        # Kombinierte Belohnung
        reward = makespan_penalty + efficiency_bonus + progress_reward + changeover_penalty

Diese kombinierte Belohnungsfunktion würde verschiedene Aspekte der Scheduling-Entscheidungen berücksichtigen und könnte zu einer besseren Makespan-Optimierung führen.

## Implementierungsdetails
### Wie der Reward in der PPO-Implementierung verwendet wird
Der berechnete Reward wird in der train_rl_agent -Funktion in main.py verwendet:

        # Aktion ausführen
        next_state, reward, done, info = env.step(action)

        # Erfahrung speichern
        agent.remember(state, action, prob, val, reward, done)

Der Reward wird zusammen mit dem Zustand, der Aktion, der Wahrscheinlichkeit der Aktion und dem Done-Flag in der Erfahrungsspeicher des Agenten gespeichert.

In der learn -Methode des PPOAgent wird der Reward dann verwendet, um die Vorteile (advantages) zu berechnen:

        def _compute_advantages(self, values, rewards, dones):
            advantages = np.zeros_like(rewards)
            returns = np.zeros_like(rewards)
            gae = 0
            
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0
                else:
                    next_value = values[t + 1]
                
                delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
                advantages[t] = gae
                returns[t] = advantages[t] + values[t]
            
            return advantages



Diese Vorteile werden dann verwendet, um die Policy (Actor) und den Value-Schätzer (Critic) zu aktualisieren:

        # Berechne Actor Loss
        dist = Categorical(dist)
        new_probs = dist.log_prob(actions_batch)
        prob_ratio = torch.exp(new_probs - old_probs_batch)
        weighted_probs = advantages_batch * prob_ratio
        clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * advantages_batch
        actor_loss = -torch.min(weighted_probs, clipped_probs).mean()

        # Berechne Critic Loss
        critic_value = torch.squeeze(critic_value)
        returns = advantages_batch + vals_arr[batch]
        critic_loss = F.mse_loss(critic_value, returns)


Durch diesen Prozess lernt der Agent, Aktionen zu wählen, die zu höheren Rewards führen, was in unserem Fall bedeutet, dass er lernt, den Makespan zu minimieren.

### Tracking des Makespans während des Trainings
Während des Trainings wird der Makespan in der train_rl_agent -Funktion verfolgt:


        # Makespan erfassen
        makespan = info.get('makespan', 0)
        all_makespans.append(makespan)

        # Bestes Modell speichern
        if makespan < best_makespan and makespan > 0:
            best_makespan = makespan
            # ... Modell speichern ...


Dies ermöglicht es, den Fortschritt des Agenten bei der Makespan-Optimierung zu verfolgen und das beste Modell zu speichern.

Mit dieser detaillierten Erklärung solltest du ein gutes Verständnis davon haben, wie der Reward-Mechanismus in deinem Job-Scheduling-System funktioniert und wie er zur Optimierung des Makespans beiträgt. Die vorgeschlagenen Optimierungen könnten dazu beitragen, die Leistung deines RL-Agenten weiter zu verbessern.