import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add the parent directory to the path to find the main module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import JobSchedulingEnvironment, PPOAgent
import numpy as np
from tqdm import tqdm


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def evaluate_scheduling_method(env, method=None, agent=None, num_episodes=10):
    """
    Evaluiert eine Scheduling-Methode (Heuristik oder trainierter Agent)
    """
    makespans = []
    total_setup_times = []
    completion_times = []
    
    for _ in tqdm(range(num_episodes), desc=f"Evaluating {'PPO' if agent else method}"):
        state = env.reset()
        done = False
        
        while not done:
            if agent:
                # PPO Agent verwenden
                action, _, _ = agent.act(state)
                state, _, done, info = env.step(action)
            else:
                # Heuristik verwenden
                state, _, done, info = env.step()
        
        # Metriken sammeln
        makespans.append(info['makespan'])
        
        # Setup-Zeiten berechnen
        total_setup = sum(
            op['setup']
            for machine_ops in env.schedule.values()
            for op in machine_ops
        )
        total_setup_times.append(total_setup)
        
        # Durchschnittliche Fertigstellungszeit der Jobs
        job_completion_times = []
        for job in env.jobs:
            job_ops = [
                op['end']
                for machine_ops in env.schedule.values()
                for op in machine_ops
                if op['job'] == job['Name']
            ]
            if job_ops:
                job_completion_times.append(max(job_ops))
        completion_times.append(np.mean(job_completion_times))
    
    return {
        'makespan': {
            'mean': np.mean(makespans),
            'std': np.std(makespans),
            'min': np.min(makespans),
            'max': np.max(makespans)
        },
        'setup_time': {
            'mean': np.mean(total_setup_times),
            'std': np.std(total_setup_times)
        },
        'completion_time': {
            'mean': np.mean(completion_times),
            'std': np.std(completion_times)
        },
        'raw_makespans': makespans
    }

def plot_comparison_results(results, save_path=None):
    """
    Visualisiert die Vergleichsergebnisse
    """
    plt.style.use('default')
    
    # Create figure with 2x2 subplots, make ax4 a polar subplot
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1], projection='polar')  # Create polar subplot
    
    # Farben für die verschiedenen Methoden
    colors = sns.color_palette("husl", len(results))
    
    # 1. Makespan-Vergleich (Boxplot)
    makespan_data = [results[method]['raw_makespans'] for method in results]
    ax1.boxplot(makespan_data, labels=results.keys())  # Fixed parameter name from tick_labels to labels
    ax1.set_title('Makespan Distribution')
    ax1.set_ylabel('Time Units')
    ax1.grid(True, alpha=0.3)
    
    # 2. Durchschnittliche Setup-Zeiten
    methods = list(results.keys())
    setup_times = [results[method]['setup_time']['mean'] for method in methods]
    setup_errors = [results[method]['setup_time']['std'] for method in methods]
    
    ax2.bar(methods, setup_times, yerr=setup_errors, capsize=5)
    ax2.set_title('Average Setup Times')
    ax2.set_ylabel('Time Units')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. Durchschnittliche Fertigstellungszeiten
    completion_times = [results[method]['completion_time']['mean'] for method in methods]
    completion_errors = [results[method]['completion_time']['std'] for method in methods]
    
    ax3.bar(methods, completion_times, yerr=completion_errors, capsize=5)
    ax3.set_title('Average Job Completion Times')
    ax3.set_ylabel('Time Units')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance-Vergleich (Radar-Chart)
    methods = list(results.keys())
    metrics = ['makespan', 'setup_time', 'completion_time']
    
    # Normalisiere die Werte für das Radar-Chart
    values = {method: [] for method in methods}
    for metric in metrics:
        metric_values = [results[method][metric]['mean'] for method in methods]
        min_val = min(metric_values)
        max_val = max(metric_values)
        for method in methods:
            normalized = (results[method][metric]['mean'] - min_val) / (max_val - min_val)
            values[method].append(normalized)
    
    # Radar-Chart erstellen
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # Schließe den Plot
    
    ax4.set_theta_zero_location('N')  # Start from top
    ax4.set_theta_direction(-1)       # Go clockwise
    plt.xticks(angles[:-1], metrics)
    
    for method, color in zip(methods, colors):
        values_method = values[method]
        values_method = np.concatenate((values_method, [values_method[0]]))
        ax4.plot(angles, values_method, color=color, label=method)
        ax4.fill(angles, values_method, color=color, alpha=0.1)
    
    ax4.set_title('Normalized Performance Comparison')
    ax4.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def print_comparison_results(results):
    """
    Gibt die Vergleichsergebnisse in tabellarischer Form aus
    """
    print("\n=== Detailed Comparison Results ===")
    print("\nMakespan Statistics:")
    print(f"{'Method':<10} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 60)
    
    for method, stats in results.items():
        print(f"{method:<10} {stats['makespan']['mean']:10.2f} {stats['makespan']['std']:10.2f} "
            f"{stats['makespan']['min']:10.2f} {stats['makespan']['max']:10.2f}")
    
    print("\nSetup Time Statistics:")
    print(f"{'Method':<10} {'Mean':>10} {'Std':>10}")
    print("-" * 35)
    
    for method, stats in results.items():
        print(f"{method:<10} {stats['setup_time']['mean']:10.2f} {stats['setup_time']['std']:10.2f}")
    
    print("\nCompletion Time Statistics:")
    print(f"{'Method':<10} {'Mean':>10} {'Std':>10}")
    print("-" * 35)
    
    for method, stats in results.items():
        print(f"{method:<10} {stats['completion_time']['mean']:10.2f} {stats['completion_time']['std']:10.2f}")


# Implementation of different scheduling methods
class SchedulingMethods:
    @staticmethod
    def fifo_schedule(env):
        """FIFO (First In, First Out) Implementation"""
        operations = []
        # Collect all operations with their arrival times
        for job in env.jobs:
            # Check if the job has 'Operations' or 'operations' key
            operations_key = 'Operationen' if 'Operationen' in job else 'operations'
            
            for op in job.get(operations_key, []):
                operations.append({
                    'job': job['Name'],
                    'operation': op,
                    'arrival_time': job.get('ArrivalTime', 0)
                })
        
        # Sort by arrival time and job ID
        operations.sort(key=lambda x: (x['arrival_time'], x['job']))
        return operations

    @staticmethod
    def lifo_schedule(env):
        """LIFO (Last In, First Out) Implementation"""
        operations = []
        # Collect all operations with their arrival times
        for job in env.jobs:
            # Check if the job has 'Operations' or 'operations' key
            operations_key = 'Operationen' if 'Operationen' in job else 'operations'
            
            for op in job.get(operations_key, []):
                operations.append({
                    'job': job['Name'],
                    'operation': op,
                    'arrival_time': job.get('ArrivalTime', 0)
                })
        
        # Sort by arrival time (reversed) and job ID (reversed)
        operations.sort(key=lambda x: (-x['arrival_time'], -int(x['job'].split('_')[1] if '_' in x['job'] else 0)))
        return operations

    @staticmethod
    def spt_schedule(env):
        """SPT (Shortest Processing Time) Implementation"""
        operations = []
        # Collect all operations with their processing times
        for job in env.jobs:
            # Check if the job has 'Operations' or 'operations' key
            operations_key = 'Operationen' if 'Operationen' in job else 'operations'
            
            for op in job.get(operations_key, []):
                processing_time_key = 'benötigteZeit' if 'benötigteZeit' in op else 'ProcessingTime'
                operations.append({
                    'job': job['Name'],
                    'operation': op,
                    'processing_time': op.get(processing_time_key, 0)
                })
        
        # Sort by processing time
        operations.sort(key=lambda x: x['processing_time'])
        return operations

    @staticmethod
    def random_schedule(env):
        """Random Scheduling Implementation"""
        operations = []
        # Collect all operations
        for job in env.jobs:
            # Check if the job has 'Operations' or 'operations' key
            operations_key = 'Operationen' if 'Operationen' in job else 'operations'
            
            for op in job.get(operations_key, []):
                operations.append({
                    'job': job['Name'],
                    'operation': op
                })
        
        # Randomly shuffle operations
        np.random.shuffle(operations)
        return operations

    @staticmethod
    def ppo_schedule(env, agent):
        """PPO Agent Implementation"""
        operations = []
        state = env.reset()
        done = False
        
        while not done:
            # Get action from PPO agent
            action, _, _ = agent.act(state)
            
            # Execute action and get next state
            next_state, _, done, info = env.step(action)
            
            # Add selected operation to schedule
            if 'selected_operation' in info:
                selected_op = info['selected_operation']
                # Check if the selected operation has the expected structure
                if isinstance(selected_op, dict) and 'operation' in selected_op:
                    operations.append(selected_op)
                else:
                    # Create proper structure if it's missing
                    job_name = selected_op.get('job', f"Job_{len(operations)}")
                    operations.append({
                        'job': job_name,
                        'operation': selected_op
                    })
            
            state = next_state
        
        return operations

def execute_schedule(env, schedule):
    """Execute a given schedule and return performance metrics and updated schedule with times"""
    makespan = 0
    total_setup_time = 0
    completion_times = []
    
    # Initialize machine timelines
    machine_times = {machine: 0 for machine in env.machines}
    job_completion = {job['Name']: 0 for job in env.jobs}
    
    # Execute each operation in the schedule
    scheduled_ops = []
    for op in schedule:
        job_name = op['job']
        operation = op['operation'].copy()  # Create a copy to modify
        
        # Bestimme die Maschine - handle beide möglichen Schlüssel
        machine_key = 'Maschine' if 'Maschine' in operation else 'Machine'
        machine = operation.get(machine_key)
        
        # Wenn keine Maschine gefunden wurde, überspringe diese Operation
        if not machine:
            print(f"Warning: No machine found for operation {operation.get('Name', 'unknown')}")
            continue
            
        # Bestimme die Verarbeitungszeit - handle beide möglichen Schlüssel
        processing_time_key = 'benötigteZeit' if 'benötigteZeit' in operation else 'ProcessingTime'
        processing_time = operation.get(processing_time_key, 0)
        
        # Calculate start time (considering machine availability and predecessors)
        predecessors_completion_time = 0
        if operation.get('Vorgänger'):
            for pred in operation['Vorgänger']:
                if pred is None:
                    continue
                pred_job = pred.split('_Op')[0]
                if pred_job in job_completion:
                    predecessors_completion_time = max(predecessors_completion_time, job_completion[pred_job])
        
        start_time = max(machine_times.get(machine, 0), job_completion.get(job_name, 0), predecessors_completion_time)
        
        # Add setup time if specified
        setup_time = operation.get('umruestkosten', 0)
        total_setup_time += setup_time
        
        # Calculate end time
        end_time = start_time + setup_time + processing_time
        
        # Update timelines
        machine_times[machine] = end_time
        job_completion[job_name] = end_time
        makespan = max(makespan, end_time)
        
        # Add timing information to operation
        operation['geplanter_Start'] = start_time
        operation['geplantes_Ende'] = end_time
        scheduled_ops.append({'job': job_name, 'operation': operation})
    
    # Calculate completion times
    completion_times = list(job_completion.values())
    
    return {
        'makespan': makespan,
        'total_setup_time': total_setup_time,
        'avg_completion_time': np.mean(completion_times) if completion_times else 0,
        'max_completion_time': max(completion_times) if completion_times else 0
    }, scheduled_ops


def save_schedule_as_json(schedule, method_name, env, parent_dir):
    """Speichert den erstellten Zeitplan im JSON-Format"""
    # Execute schedule to get timing information
    metrics, scheduled_ops = execute_schedule(env, schedule)
    
    # Group operations by job
    scheduled_jobs = {}
    for op in scheduled_ops:
        job_name = op['job']
        if job_name not in scheduled_jobs:
            # Find original job to get priority
            original_job = next((job for job in env.jobs if job['Name'] == job_name), None)
            scheduled_jobs[job_name] = {
                "Name": job_name,
                "Priorität": original_job.get('Priorität', 1) if original_job else 1,
                "Operationen": []
            }
        
        # Add operation with timing information
        scheduled_jobs[job_name]["Operationen"].append(op['operation'])
    
    # Create output data
    output_data = {
        "scheduling_method": method_name,
        "makespan": metrics['makespan'],
        "total_setup_time": metrics['total_setup_time'],
        "jobs": list(scheduled_jobs.values())
    }
    
    # Save to file
    output_file = os.path.join(parent_dir, f"output_{method_name.lower()}_schedule.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
    
    print(f"✓ Saved {method_name} schedule to {output_file}")
    return output_file
def main():
    # Define absolute paths for data and model files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_file = os.path.join(parent_dir, "production_data.json")
    
    # Use a specific model instead of finding the best one
    specific_model = "best_model_makespan_2079.00.pt"
    model_path = os.path.join(parent_dir, "model_checkpoints", specific_model)  # Removed duplicate '040425'
    
    print(f"Looking for data file at: {data_file}")
    print(f"Using specific model from: {model_path}")
    
    # Set number of evaluation episodes
    num_evaluation_episodes = 1
    
    # Generate new data with 50 jobs to match the model's training conditions
    from main import generate_synthetic_data, save_data
    
    config = {
        "n_jobs": 50,
        "min_ops": 1,
        "max_ops": 5,
        "machines": ["M1", "M2", "M3", "M4"],
        "materials": ["Material_A", "Material_B", "Material_C"],
        "tools": ["Kühlmittel", "Öl", "Werkzeug", "Schablone"]
    }
    
    # Generate data with the same configuration as training
    print("Generating new test data with 50 jobs...")
    test_data = generate_synthetic_data(config)
    test_data_file = os.path.join(parent_dir, "test_data_50jobs.json")
    save_data(test_data, test_data_file)
    
    # Use the new test data file for evaluation
    data_file = test_data_file
    
    # Lade das trainierte PPO-Modell
    env = JobSchedulingEnvironment(data_file)
    state = env.reset()
    state_dim = state['x'].shape[1]
    action_dim = len(state['operation_queue'])
    
    # Modell mit den gleichen Hyperparametern wie beim Training erstellen
    agent = PPOAgent(state_dim, action_dim, hidden_dim=128, num_heads=8)
    agent.load(model_path)
    print("✓ Loaded PPO model from", model_path)
    
    # Rest des Codes bleibt unverändert
    methods = {
        'FIFO': None,
        'LIFO': None,
        'SPT': None,
        'Random': None,
        'PPO': agent
    }
    
    results = {}
    
    # First, evaluate using the existing evaluation method
    for method_name, method_agent in methods.items():
        print(f"\nEvaluating {method_name}...")
        env = JobSchedulingEnvironment(data_file, heuristic=method_name if method_agent is None else None)
        results[method_name] = evaluate_scheduling_method(
            env, 
            method=method_name if method_agent is None else None,
            agent=method_agent,
            num_episodes=num_evaluation_episodes
        )
    
    # Add detailed scheduling analysis
    print("\nPerforming detailed scheduling analysis...")
    scheduler = SchedulingMethods()
    env = JobSchedulingEnvironment(data_file)
    
    # Get schedule using different methods
    detailed_results = {}
    
    # Fix: Unpack the tuple returned by execute_schedule
    metrics_fifo, _ = execute_schedule(env, scheduler.fifo_schedule(env))
    detailed_results['FIFO'] = metrics_fifo
    
    metrics_lifo, _ = execute_schedule(env, scheduler.lifo_schedule(env))
    detailed_results['LIFO'] = metrics_lifo
    
    metrics_spt, _ = execute_schedule(env, scheduler.spt_schedule(env))
    detailed_results['SPT'] = metrics_spt
    
    metrics_random, _ = execute_schedule(env, scheduler.random_schedule(env))
    detailed_results['Random'] = metrics_random
    
    metrics_ppo, _ = execute_schedule(env, scheduler.ppo_schedule(env, agent))
    detailed_results['PPO'] = metrics_ppo
    
    # Print detailed scheduling results
    print("\n=== Detailed Scheduling Analysis ===")
    for method, metrics in detailed_results.items():
        print(f"\n{method} Scheduling Metrics:")
        print(f"Makespan: {metrics['makespan']:.2f}")
        print(f"Total Setup Time: {metrics['total_setup_time']:.2f}")
        print(f"Average Completion Time: {metrics['avg_completion_time']:.2f}")
        print(f"Maximum Completion Time: {metrics['max_completion_time']:.2f}")
    
    # Continue with existing result visualization
    print_comparison_results(results)
    plot_comparison_results(results, save_path='scheduling_comparison_50jobs.png')
    
    # Beste Methode identifizieren
    best_method = min(results.items(), key=lambda x: x[1]['makespan']['mean'])
    print(f"\nBest performing method: {best_method[0]}")
    print(f"Average makespan: {best_method[1]['makespan']['mean']:.2f}")
    
    # Verbesserung gegenüber Baseline (FIFO) berechnen
    fifo_makespan = results['FIFO']['makespan']['mean']
    best_makespan = best_method[1]['makespan']['mean']
    improvement = (fifo_makespan - best_makespan) / fifo_makespan * 100
    print(f"Improvement over FIFO: {improvement:.2f}%")
    
    # Erstelle JSON-Dateien für jede Methode
    print("\nSaving results to JSON files...")
    for method_name in methods.keys():
        # Kombiniere die Ergebnisse aus beiden Evaluierungen
        combined_results = {
            # Ergebnisse aus der Hauptevaluierung
            "evaluation_metrics": results[method_name],
            # Ergebnisse aus der detaillierten Analyse
            "detailed_metrics": detailed_results[method_name] if method_name in detailed_results else {}
        }
        
        # Füge Informationen über die Umgebung hinzu
        combined_results["environment_info"] = {
            "num_jobs": len(env.jobs),
            "num_machines": len(env.machines),
            "data_file": data_file
        }
        
        # Speichere die Ergebnisse in einer JSON-Datei
        result_file = os.path.join(parent_dir, f"results_{method_name.lower()}_50jobs.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(combined_results, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
        print(f"✓ Saved {method_name} results to {result_file}")

    # Erstelle zusätzlich die Zeitplan-JSON-Dateien im Format ähnlich der Eingabedaten
    print("\nSaving schedules in input-like format...")
    
    # Erstelle für jede Methode einen Zeitplan und speichere ihn
    scheduler = SchedulingMethods()
    env = JobSchedulingEnvironment(data_file)
    
    # FIFO Schedule
    fifo_schedule = scheduler.fifo_schedule(env)
    save_schedule_as_json(fifo_schedule, "FIFO", env, parent_dir)
    
    # LIFO Schedule
    lifo_schedule = scheduler.lifo_schedule(env)
    save_schedule_as_json(lifo_schedule, "LIFO", env, parent_dir)
    
    # SPT Schedule
    spt_schedule = scheduler.spt_schedule(env)
    save_schedule_as_json(spt_schedule, "SPT", env, parent_dir)
    
    # Random Schedule
    random_schedule = scheduler.random_schedule(env)
    save_schedule_as_json(random_schedule, "Random", env, parent_dir)
    
    # PPO Schedule
    ppo_schedule = scheduler.ppo_schedule(env, agent)
    save_schedule_as_json(ppo_schedule, "PPO", env, parent_dir)
    
    print("\nAll schedules saved successfully!")

if __name__ == "__main__":
    main()

