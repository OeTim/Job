import gym
import numpy as np
from gym import spaces
import random
import copy
from collections import deque
import math

class JobSchedulingEnv(gym.Env):
    """
    Job Scheduling Environment für Reinforcement Learning
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, data, config=None):
        super(JobSchedulingEnv, self).__init__()
        
        self.data = data
        self.config = config
        self.jobs = data["jobs"]
        self.n_jobs = len(self.jobs)
        self.n_machines = len(set([op["Maschine"] for job in self.jobs for op in job["Operationen"]]))
        
        # Zustand: [Wartende Jobs, Maschinenstatus, Zeit]
        self.observation_space = spaces.Dict({
            'waiting_jobs': spaces.Box(low=0, high=1, shape=(self.n_jobs, 10), dtype=np.float32),
            'machine_status': spaces.Box(low=0, high=1, shape=(self.n_machines, 3), dtype=np.float32),
            'time_features': spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        })
        
        # Aktionen: Job-Auswahl für die nächste Ausführung
        self.action_space = spaces.Discrete(self.n_jobs)
        
        # Scheduling-Strategien
        self.strategies = {
            'FIFO': self._fifo_schedule,
            'LIFO': self._lifo_schedule,
            'SPT': self._spt_schedule,
            'RANDOM': self._random_schedule
        }

        self.machine_materials = self.config.get("machine_initial", {})
        
        self.reset()
    
    def reset(self):
        """Umgebung zurücksetzen"""
        # Tiefe Kopie der Jobs erstellen
        self.remaining_jobs = copy.deepcopy(self.jobs)
        self.completed_jobs = []
        self.current_time = 0
        self.machine_availability = {machine: 0 for machine in 
                                    set([op["Maschine"] for job in self.jobs for op in job["Operationen"]])}
        self.makespan = 0
        self.job_completion_times = {}
        self.scheduled_operations = []
        
        # Warteschlange für jede Maschine
        self.machine_queues = {machine: deque() for machine in self.machine_availability.keys()}
        
        # Materialzustand der Maschinen zurücksetzen
        self.machine_materials = copy.deepcopy(self.config.get("machine_initial", {}))
        
        # Operationen, die bereit zur Ausführung sind
        self.ready_operations = self._get_ready_operations()
        
        return self._get_observation()
    
    def step(self, action):
        """Führt einen Schritt in der Umgebung aus"""
        # ... existing code ...
        
        # Belohnung berechnen (verbesserte Version)
        prev_makespan = self.makespan
        new_makespan = max(self.makespan, completion_time)
        makespan_diff = new_makespan - prev_makespan
        
        # Maschinenauslastung berechnen
        machine_times = list(self.machine_availability.values())
        max_time = max(machine_times) if machine_times else 0
        min_time = min(machine_times) if machine_times else 0
        machine_balance = 1.0 - ((max_time - min_time) / (max_time + 1e-6))  # Ausgewogenheit der Maschinenauslastung
        
        # Kritischer Pfad Analyse
        # Identifiziere Operationen auf dem kritischen Pfad (längster Pfad im Schedule)
        critical_path_bonus = 0
        if self.ready_operations:
            # Finde Operationen, die zur aktuellen Zeit den längsten verbleibenden Pfad haben
            remaining_times = []
            for op in self.ready_operations:
                job_name = op['job']
                job = next((j for j in self.remaining_jobs if j['Name'] == job_name), None)
                if job:
                    # Berechne die verbleibende Bearbeitungszeit für diesen Job
                    remaining_time = sum(o['processing_time'] for o in self.ready_operations if o['job'] == job_name)
                    remaining_times.append(remaining_time)
            
            # Wenn die gewählte Operation auf dem kritischen Pfad liegt, gib einen Bonus
            if remaining_times and action < len(remaining_times):
                if remaining_times[action] == max(remaining_times):
                    critical_path_bonus = 10.0
        
        # Umrüstzeit-Optimierung
        changeover_penalty = 0
        if len(self.scheduled_operations) > 0:
            last_op = self.scheduled_operations[-1]
            current_op = self.ready_operations[action] if action < len(self.ready_operations) else None
            
            if current_op and last_op['machine'] == current_op['machine']:
                # Prüfe, ob ein Materialwechsel stattfindet
                last_material = None
                current_material = None
                
                # Finde das Material der letzten Operation
                for job in self.jobs:
                    for op in job['Operationen']:
                        if op['Name'] == last_op['operation']:
                            if 'produziertesMaterial' in op:
                                last_material = op['produziertesMaterial']
                
                # Finde das Material der aktuellen Operation
                job_name = current_op['job']
                op_name = current_op['operation']
                for job in self.jobs:
                    if job['Name'] == job_name:
                        for op in job['Operationen']:
                            if op['Name'] == op_name:
                                if 'produziertesMaterial' in op:
                                    current_material = op['produziertesMaterial']
                
                # Wenn beide Materialien bekannt sind und unterschiedlich, berechne die Umrüstzeit
                if last_material and current_material and last_material != current_material:
                    changeover_times = self.config.get("changeover_times", {}).get(current_op['machine'], {})
                    changeover_time = changeover_times.get((last_material, current_material), 0)
                    changeover_penalty = -changeover_time * 0.5  # Negative Belohnung für Umrüstzeit
        
        # Fortschrittsfaktor mit Gewichtung für verbleibende Jobs
        progress = len(self.completed_jobs) / len(self.jobs)
        remaining_ratio = len(self.remaining_jobs) / len(self.jobs)
        progress_weight = 1.0 + (1.0 - remaining_ratio) * 2.0  # Höheres Gewicht gegen Ende
        
        # Kombinierte Belohnung mit stärkerer Differenzierung
        r_makespan = -makespan_diff * 2.0  # Erhöhte Gewichtung für Makespan-Reduktion
        r_balance = machine_balance * 8.0
        r_critical = critical_path_bonus
        r_changeover = changeover_penalty
        r_progress = progress * 15.0 * progress_weight
        
        # Adaptive Komponente: Belohne stärker, wenn wir uns dem Ende nähern
        completion_factor = (len(self.completed_jobs) / len(self.jobs)) ** 2
        adaptive_bonus = completion_factor * 20.0
        
        # Gewichtete Summe
        reward = r_makespan + r_balance + r_critical + r_changeover + r_progress + adaptive_bonus
        
        # Zusätzlicher Bonus für Fertigstellung aller Jobs
        if len(self.remaining_jobs) == 0:
            reward += 100.0
        
        # Prüfen, ob alle Jobs abgeschlossen sind
        done = len(self.remaining_jobs) == 0
        
        # Neue ready operations bestimmen
        self.ready_operations = self._get_ready_operations()
        
        return self._get_observation(), reward, done, {"makespan": self.makespan}
    
    def _execute_job(self, job):
        """Führt einen Job aus und gibt die Fertigstellungszeit zurück"""
        job_completion_time = 0
        
        for op in job["Operationen"]:
            machine = op["Maschine"]
            processing_time = op["benötigteZeit"] 
            
            # Frühester Startzeitpunkt ist die Verfügbarkeit der Maschine
            start_time = max(self.current_time, self.machine_availability[machine])
            
            if "produziertesMaterial" in op:
                required_material = op["produziertesMaterial"]
                current_material = self.machine_materials.get(machine)
                
                if current_material != required_material and current_material is not None:
                    # Hole die Umrüstzeit aus der Tabelle
                    changeover_times = self.config.get("changeover_times", {}).get(machine, {})
                    changeover_time = changeover_times.get((current_material, required_material), 0)
                    
                    # Umrüstzeit zum Startzeitpunkt hinzufügen
                    start_time += changeover_time
                
                # Aktualisiere den Materialzustand der Maschine
                self.machine_materials[machine] = required_material
            
            # Operation ausführen
            end_time = start_time + processing_time
            
            # Zwischenlagerzeit hinzufügen, falls vorhanden
            if "zwischenlager" in op:
                end_time += op["zwischenlager"]["minVerweildauer"]
            
            # Maschine aktualisieren
            self.machine_availability[machine] = end_time
            
            # Operation zur Liste der geplanten Operationen hinzufügen
            self.scheduled_operations.append({
                "job": job["Name"],
                "operation": op["Name"],
                "machine": machine,
                "start": start_time,
                "end": end_time
            })
            
            # Fertigstellungszeit des Jobs aktualisieren
            job_completion_time = max(job_completion_time, end_time)
        
        return job_completion_time
    
    def _get_ready_operations(self):
        """Bestimmt Operationen, die bereit zur Ausführung sind"""
        ready_ops = []
        
        for job in self.remaining_jobs:
            for op in job["Operationen"]:
                # Prüfen, ob alle Vorgänger abgeschlossen sind
                predecessors_completed = True
                if op["Vorgänger"]:
                    for pred in op["Vorgänger"]:
                        if not any(sched_op["operation"] == pred for sched_op in self.scheduled_operations):
                            predecessors_completed = False
                            break
                
                if predecessors_completed:
                    ready_ops.append({
                        "job": job["Name"],
                        "operation": op["Name"],
                        "processing_time": op["benötigteZeit"],
                        "machine": op["Maschine"]
                    })
        
        return ready_ops
    
    def _get_observation(self):
        """Erstellt den Beobachtungsvektor für den aktuellen Zustand"""
        # Features für wartende Jobs
        waiting_jobs = np.zeros((self.n_jobs, 10), dtype=np.float32)
        for i, job in enumerate(self.remaining_jobs):
            if i >= self.n_jobs:
                break
                
            # Job-Features
            waiting_jobs[i, 0] = job.get("Priorität", 5) / 10  # Normalisierte Priorität
            
            # Operationen-Features
            total_time = sum(op["benötigteZeit"] for op in job["Operationen"])
            waiting_jobs[i, 1] = total_time / 300  # Normalisierte Gesamtzeit
            waiting_jobs[i, 2] = len(job["Operationen"]) / 5  # Normalisierte Anzahl Operationen
            
            # Durchschnittliche Bearbeitungszeit
            avg_time = total_time / len(job["Operationen"]) if job["Operationen"] else 0
            waiting_jobs[i, 3] = avg_time / 60  # Normalisiert
            
            # Anzahl benötigter Werkzeuge
            tools_count = sum(1 for op in job["Operationen"] if "benötigteHilfsmittel" in op)
            waiting_jobs[i, 4] = tools_count / len(job["Operationen"]) if job["Operationen"] else 0

            setup_times = 0
            for op in job["Operationen"]:
                if "produziertesMaterial" in op:
                    machine = op["Maschine"]
                    required_material = op["produziertesMaterial"]
                    current_material = self.machine_materials.get(machine)
                    
                    if current_material != required_material and current_material is not None:
                        changeover_times = self.config.get("changeover_times", {}).get(machine, {})
                        setup_times += changeover_times.get((current_material, required_material), 0)
            
            waiting_jobs[i, 5] = setup_times / 50  # Normalisiert

            
            # Zwischenlagerzeiten
            storage_times = sum(op.get("zwischenlager", {}).get("minVerweildauer", 0) for op in job["Operationen"])
            waiting_jobs[i, 6] = storage_times / 100  # Normalisiert
            
            # Restliche Features für zukünftige Erweiterungen
            waiting_jobs[i, 7:10] = 0
        
        # Features für Maschinenstatus
        machine_status = np.zeros((self.n_machines, 3), dtype=np.float32)
        for i, (machine, avail_time) in enumerate(self.machine_availability.items()):
            if i >= self.n_machines:
                break
                
            # Verfügbarkeitszeit
            machine_status[i, 0] = avail_time / 1000  # Normalisiert
            
            # Warteschlangenlänge
            queue_length = len(self.machine_queues.get(machine, []))
            machine_status[i, 1] = queue_length / 10  # Normalisiert
            
            # Auslastung (Verhältnis von Nutzungszeit zu Gesamtzeit)
            if self.current_time > 0:
                utilization = (avail_time - self.current_time) / self.current_time
                machine_status[i, 2] = min(1.0, utilization)  # Auf [0,1] begrenzen
        
        # Zeitbezogene Features
        time_features = np.zeros(3, dtype=np.float32)
        time_features[0] = self.current_time / 1000  # Normalisierte aktuelle Zeit
        time_features[1] = self.makespan / 1000  # Normalisierter Makespan
        time_features[2] = len(self.completed_jobs) / self.n_jobs  # Fortschritt
        
        return {
            'waiting_jobs': waiting_jobs,
            'machine_status': machine_status,
            'time_features': time_features
        }
    
    def render(self, mode='human'):
        """Visualisiert den aktuellen Zustand"""
        if mode == 'human':
            print(f"Zeit: {self.current_time}, Makespan: {self.makespan}")
            print(f"Abgeschlossene Jobs: {len(self.completed_jobs)}/{self.n_jobs}")
            print("Maschinenverfügbarkeit:")
            for machine, time in self.machine_availability.items():
                print(f"  {machine}: {time}")
    
    def _fifo_schedule(self):
        """First In, First Out Scheduling"""
        return 0  # Nimm den ersten Job in der Liste
    
    def _lifo_schedule(self):
        """Last In, First Out Scheduling"""
        return len(self.remaining_jobs) - 1  # Nimm den letzten Job in der Liste
    
    def _spt_schedule(self):
        """Shortest Processing Time Scheduling"""
        if not self.remaining_jobs:
            return 0
            
        # Berechne Gesamtbearbeitungszeit für jeden Job
        processing_times = []
        for job in self.remaining_jobs:
            total_time = sum(op["benötigteZeit"] for op in job["Operationen"])
            processing_times.append(total_time)
        
        # Wähle Job mit kürzester Bearbeitungszeit
        return np.argmin(processing_times)

    def _random_schedule(self):
        """Zufällige Job-Auswahl"""
        if not self.remaining_jobs:
            return 0
        return random.randint(0, len(self.remaining_jobs) - 1)