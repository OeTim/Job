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
        self.total_operations = sum(len(job["Operationen"]) for job in self.jobs)
        
        # Aktionen: Job-Auswahl für die nächste Ausführung
        self.action_space = spaces.Discrete(self.total_operations)
        
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
        
        # Initialisiere job_availability
        self.job_availability = {job["Name"]: 0 for job in self.jobs}
        
        # Warteschlange für jede Maschine
        self.machine_queues = {machine: deque() for machine in self.machine_availability.keys()}
        
        # Materialzustand der Maschinen zurücksetzen
        self.machine_materials = copy.deepcopy(self.config.get("machine_initial", {}))
        
        # Operationen, die bereit zur Ausführung sind
        self.ready_operations = self._get_ready_operations()
        
        return self._get_observation()
    
    def step(self, action):
        # Handle strategy strings
        if isinstance(action, str):
            if action in self.strategies:
                action = self.strategies[action]()
            else:
                raise ValueError(f"Unknown strategy: {action}")
        
        # Validate action
        if not self.ready_operations or action >= len(self.ready_operations):
            return self._get_observation(), -10, True, {"makespan": self.makespan}
        
        # Execute selected operation
        selected_op = self.ready_operations[action]
        job_name = selected_op['job']
        op_name = selected_op['operation']
        machine = selected_op['machine']
        processing_time = selected_op['processing_time']
        
        # Determine start time
        machine_available = self.machine_availability.get(machine, 0)
        job_available = self.job_availability.get(job_name, 0)
        start_time = max(machine_available, job_available)
        
        # Calculate changeover time
        current_material = self.machine_materials.get(machine)
        required_material = None
        for job in self.remaining_jobs:
            if job['Name'] == job_name:
                for op in job['Operationen']:
                    if op['Name'] == op_name and 'produziertesMaterial' in op:
                        required_material = op['produziertesMaterial']
                        break
                break
        
        changeover_time = 0
        if current_material and required_material and current_material != required_material:
            changeover_times = self.config.get("changeover_times", {}).get(machine, {})
            changeover_time = changeover_times.get((current_material, required_material), 0)
        
        # Update start and end times
        start_time += changeover_time
        end_time = start_time + processing_time
        
        # Update machine and job status
        self.machine_availability[machine] = end_time
        self.job_availability[job_name] = end_time
        if required_material:
            self.machine_materials[machine] = required_material
        
        # Record scheduled operation
        self.scheduled_operations.append({
            'job': job_name,
            'operation': op_name,
            'machine': machine,
            'start': start_time,
            'end': end_time
        })
        
        # Calculate total operations and completion status
        total_ops = sum(len(job["Operationen"]) for job in self.jobs)
        completed_ops = len(self.scheduled_operations)
        remaining_ops = total_ops - completed_ops
        
        # Update job status and check completion
        for job in self.remaining_jobs:
            if job['Name'] == job_name:
                # Find and mark completed operation
                for op in job['Operationen']:
                    if op['Name'] == op_name:
                        if 'completed_operations' not in job:
                            job['completed_operations'] = []
                        job['completed_operations'].append(op)
                        
                # Check if all operations in job are completed
                all_completed = all(
                    any(sched_op['operation'] == op['Name'] 
                        for sched_op in self.scheduled_operations)
                    for op in job['Operationen']
                )
                
                if all_completed:
                    self.completed_jobs.append(job)
                    self.remaining_jobs.remove(job)
                break
        
        # Calculate rewards
        operations_penalty = -100 * (remaining_ops / total_ops)
        
        efficiency_bonus = 0
        if processing_time > 0:
            efficiency = 1.0 / (processing_time + changeover_time)
            efficiency_bonus = efficiency * 20
        
        utilization_bonus = 0
        if self.current_time > 0:
            utilization = sum(self.machine_availability.values()) / (self.n_machines * self.current_time)
            utilization_bonus = utilization * 30
        
        progress_reward = (len(self.completed_jobs) / self.n_jobs) * 50
        
        changeover_penalty = -changeover_time / 5 if changeover_time > 0 else 0
        
        waiting_time_penalty = -(start_time - self.current_time) / 50
        
        # Calculate final reward
        reward = (operations_penalty +
                 efficiency_bonus +
                 utilization_bonus +
                 progress_reward +
                 changeover_penalty +
                 waiting_time_penalty)
        
        # Add completion bonuses
        if completed_ops == total_ops:
            reward += 500  # Bonus for completing all operations
        elif len(self.completed_jobs) > 0:
            reward += 50  # Bonus for completing any job
        
        # Update makespan and check completion
        self.makespan = max(self.makespan, end_time)
        done = len(self.remaining_jobs) == 0
        
        # Get next available operations
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
            for op in job['Operationen']:
                # Überprüfen, ob die Operation bereits geplant wurde
                if any(sched_op['operation'] == op['Name'] for sched_op in self.scheduled_operations):
                    continue
                
                # Prüfen, ob alle Vorgänger abgeschlossen sind
                # Überprüfe nur die Vorgängerbedingungen
                predecessors_completed = True
                if op['Vorgänger']:
                    for pred in op['Vorgänger']:
                        if not any(sched_op['operation'] == pred for sched_op in self.scheduled_operations):
                            predecessors_completed = False
                            break
                
                if predecessors_completed:
                    ready_ops.append({
                        'job': job['Name'],
                        'operation': op['Name'],
                        'processing_time': op['benötigteZeit'],
                        'machine': op['Maschine']
                    })
        
        return ready_ops
    
    def _get_observation(self):
        """Erstellt den Beobachtungsvektor für den aktuellen Zustand"""
        # Calculate maximum possible values for normalization
        max_processing_time = 60  # Typical max processing time
        max_possible_time = 300   # Typical max total time
        current_makespan = max(1, self.makespan)  # Avoid division by zero
        
        # Features für wartende Jobs
        waiting_jobs = np.zeros((self.n_jobs, 10), dtype=np.float32)
        for i, job in enumerate(self.remaining_jobs):
            if i >= self.n_jobs:
                break
                
            # Job-Features
            waiting_jobs[i, 0] = job.get("Priorität", 5) / 10  # Normalisierte Priorität
            
            # Operationen-Features
            total_time = sum(op["benötigteZeit"] for op in job["Operationen"])
            waiting_jobs[i, 1] = total_time / max_possible_time  # Dynamische Normalisierung
            waiting_jobs[i, 2] = len(job["Operationen"]) / 5  # Normalisierte Anzahl Operationen
            
            # Durchschnittliche Bearbeitungszeit
            avg_time = total_time / len(job["Operationen"]) if job["Operationen"] else 0
            waiting_jobs[i, 3] = avg_time / max_processing_time  # Dynamische Normalisierung
            
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
            machine_status[i, 0] = avail_time / current_makespan if current_makespan > 0 else 0  # Relative Normalisierung
            
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
        return min(range(len(self.ready_operations)), 
                key=lambda i: self.ready_operations[i]['processing_time'])

    def _random_schedule(self):
        return random.randint(0, len(self.ready_operations) - 1)