import simpy
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from environments.job_env import JobSchedulingEnv
from src.utils.config import config as default_config  # Import default config

def simulate_production(data, config=None, strategy=None, random_seed=42):
    """
    Simuliert die Produktion mit SimPy
    
    Args:
        data: Produktionsdaten
        config: Konfiguration für die Simulation
        strategy: Scheduling-Strategie (None, 'FIFO', 'LIFO', 'SPT')
        random_seed: Seed für Zufallszahlen
    """
    # Use provided config or default config
    if config is None:
        config = default_config
        
    random.seed(random_seed)
    env = simpy.Environment()
    machine_states = config.get("machine_initial", {machine: None for machine in config["machines"]})
    
    # Ressourcen erstellen
    machines = {machine: simpy.Resource(env, capacity=1) for machine in config["machines"]}
    tools = {tool: simpy.Resource(env, capacity=3) for tool in config["tools"]}
    
    # Statistiken
    stats = {
        "job_completion_times": {},
        "machine_utilization": {machine: 0 for machine in machines},
        "tool_utilization": {tool: 0 for tool in tools},
        "waiting_times": [],
        "scheduled_operations": []  # Liste der geplanten Operationen für Gantt-Diagramm
    }
    
    # Prozess für einen Job
    def job_process(env, job):
        job_name = job["Name"]
        start_time = env.now
        
        for op in job["Operationen"]:
            op_name = op["Name"]
            machine_name = op["Maschine"]
            
            # Warten auf Maschine
            wait_start = env.now
            with machines[machine_name].request() as req:
                yield req
                wait_time = env.now - wait_start
                stats["waiting_times"].append(wait_time)
                
                # Werkzeuge anfordern, falls benötigt
                tool_reqs = []
                if "benötigteHilfsmittel" in op:
                    for tool in op["benötigteHilfsmittel"]:
                        tool_reqs.append(tools[tool].request())
                    
                    # Auf alle Werkzeuge warten
                    for req in tool_reqs:
                        yield req
                
                # # Umrüstzeit, falls vorhanden
                # if "umruestzeit" in op:
                #     yield env.timeout(op["umruestzeit"])
                
                required_material = op["produziertesMaterial"]
                current_material = machine_states[machine_name]
                if current_material != required_material:
                    # Hole die Umrüstzeit aus der Tabelle
                    changeover_table = config["changeover_times"].get(machine_name, {})
                    # Falls current_material None ist, kann man optional keinen Wechsel oder einen definierten Initialwert nehmen
                    changeover_time = changeover_table.get((current_material, required_material), 0)
                    if changeover_time > 0:
                        print(f"Maschine {machine_name} wechselt von {current_material} zu {required_material}. Umrüstzeit: {changeover_time}")
                        yield env.timeout(changeover_time)
                        # OPTIONAL: Hier könnte man einen Strafwert (Penalty) für die Umrüstung einbauen, z.B.:
                        # stats["umruest_penalties"].append(changeover_time)
                    # Aktualisiere den Maschinenzustand
                    machine_states[machine_name] = required_material

                # Operation ausführen
                operation_start = env.now
                yield env.timeout(op["benötigteZeit"])
                operation_end = env.now
                
                # Operation zur Liste der geplanten Operationen hinzufügen
                stats["scheduled_operations"].append({
                    "job": job_name,
                    "operation": op_name,
                    "machine": machine_name,
                    "start": operation_start,
                    "end": operation_end
                })
                
                # Maschinennutzung aktualisieren
                stats["machine_utilization"][machine_name] += op["benötigteZeit"]
                
                # Werkzeuge freigeben
                for i, tool in enumerate(op.get("benötigteHilfsmittel", [])):
                    stats["tool_utilization"][tool] += op["benötigteZeit"]
                    tools[tool].release(tool_reqs[i])
                
                # Zwischenlagerung, falls erforderlich
                if "zwischenlager" in op:
                    yield env.timeout(op["zwischenlager"]["minVerweildauer"])
        
        # Job abgeschlossen
        completion_time = env.now - start_time
        stats["job_completion_times"][job_name] = completion_time
    
    # Jobs nach Strategie sortieren
    jobs = data["jobs"].copy()
    if strategy == 'FIFO':
        # FIFO: Keine Änderung der Reihenfolge
        pass
    elif strategy == 'LIFO':
        # LIFO: Umgekehrte Reihenfolge
        jobs.reverse()
    elif strategy == 'SPT':
        # SPT: Sortieren nach Gesamtbearbeitungszeit
        jobs.sort(key=lambda job: sum(op["benötigteZeit"] for op in job["Operationen"]))
    elif strategy == 'RANDOM':
        # RANDOM: Zufällige Reihenfolge
        random.shuffle(jobs)
    
    # Alle Jobs starten
    for job in jobs:
        env.process(job_process(env, job))
    
    # Simulation ausführen
    env.run()
    
    # Makespan berechnen
    makespan = max(stats["job_completion_times"].values()) if stats["job_completion_times"] else 0
    stats["makespan"] = makespan
    
    return stats


def compare_strategies(data, config=None, include_rl=False, rl_agent=None, n_runs=5):
    """Compares different scheduling strategies"""
    # Import JobSchedulingEnv if not already imported
    from environments.job_env import JobSchedulingEnv
    
    strategies = ['FIFO', 'LIFO', 'SPT', 'RANDOM']
    results = {}
    
    env = JobSchedulingEnv(data, config)
    
    # Strategien testen
    for strategy in strategies:
        if strategy == 'RANDOM':
            # Mehrere Durchläufe für die Zufallsstrategie
            makespans = []
            for _ in range(n_runs):
                env.reset()
                done = False
                while not done:
                    _, _, done, info = env.step('RANDOM')
                makespans.append(info['makespan'])
            
            # Durchschnitt, Minimum und Maximum berechnen
            results[strategy] = {
                'avg_makespan': np.mean(makespans),
                'min_makespan': np.min(makespans),
                'max_makespan': np.max(makespans),
                'std_makespan': np.std(makespans)
            }
            print(f"\n{strategy} (über {n_runs} Durchläufe):")
            print(f"  Durchschnittlicher Makespan: {results[strategy]['avg_makespan']:.2f}")
            print(f"  Minimaler Makespan: {results[strategy]['min_makespan']:.2f}")
            print(f"  Maximaler Makespan: {results[strategy]['max_makespan']:.2f}")
            print(f"  Standardabweichung: {results[strategy]['std_makespan']:.2f}")
        else:
            # Einzelner Durchlauf für deterministische Strategien
            env.reset()
            done = False
            while not done:
                _, _, done, info = env.step(strategy)
            
            results[strategy] = {'makespan': info['makespan']}
            print(f"\n{strategy}:")
            print(f"  Makespan: {info['makespan']:.2f}")
    
    # RL-Agent testen, falls gewünscht
    # Create environment for RL agent if needed
    if include_rl and rl_agent is not None:
        env = JobSchedulingEnv(data, config)
        makespans = []
        for _ in range(n_runs):
            state = env.reset()
            done = False
            while not done:
                # Deaktiviere Exploration für die Evaluation
                with torch.no_grad():
                    action, _, _ = rl_agent.choose_action(state)
                state, _, done, info = env.step(action)
            makespans.append(info['makespan'])
        
        results['RL'] = {
            'avg_makespan': np.mean(makespans),
            'min_makespan': np.min(makespans),
            'max_makespan': np.max(makespans),
            'std_makespan': np.std(makespans)
        }
        print(f"\nRL-Agent (über {n_runs} Durchläufe):")
        print(f"  Durchschnittlicher Makespan: {results['RL']['avg_makespan']:.2f}")
        print(f"  Minimaler Makespan: {results['RL']['min_makespan']:.2f}")
        print(f"  Maximaler Makespan: {results['RL']['max_makespan']:.2f}")
        print(f"  Standardabweichung: {results['RL']['std_makespan']:.2f}")
    

    
    # Visualisierung der Ergebnisse
    plt.figure(figsize=(10, 6))
    
    # Balkendiagramm für deterministische Strategien
    deterministic_strategies = [s for s in strategies if s != 'RANDOM']
    deterministic_makespans = [results[s]['makespan'] for s in deterministic_strategies]
    
    # Balken für deterministische Strategien
    plt.bar(deterministic_strategies, deterministic_makespans, color='blue', alpha=0.7)
    
    # Balken für Zufallsstrategie mit Fehlerbalken
    random_pos = len(deterministic_strategies)
    plt.bar(random_pos, results['RANDOM']['avg_makespan'], color='orange', alpha=0.7)
    plt.errorbar(random_pos, results['RANDOM']['avg_makespan'], 
                yerr=results['RANDOM']['std_makespan'], fmt='o', color='red')
    
    # Balken für RL-Agent, falls vorhanden
    if include_rl and rl_agent is not None:
        rl_pos = random_pos + 1
        plt.bar(rl_pos, results['RL']['avg_makespan'], color='green', alpha=0.7)
        plt.errorbar(rl_pos, results['RL']['avg_makespan'], 
                    yerr=results['RL']['std_makespan'], fmt='o', color='red')
        plt.xticks(range(len(deterministic_strategies) + 2), deterministic_strategies + ['RANDOM', 'RL'])
    else:
        plt.xticks(range(len(deterministic_strategies) + 1), deterministic_strategies + ['RANDOM'])
    
    plt.ylabel('Makespan')
    plt.title('Vergleich der Scheduling-Strategien')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    return results
