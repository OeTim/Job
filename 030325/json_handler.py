import json
import os
import random
from typing import Dict, List, Optional

def load_jobs_json(file_path: str) -> Dict:
    """
    Lädt die JSON-Datei mit den Job-Informationen.
    
    Args:
        file_path: Pfad zur JSON-Datei
        
    Returns:
        Dictionary mit den Job-Daten
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Fehler: Die Datei {file_path} wurde nicht gefunden.")
        return {}
    except json.JSONDecodeError:
        print(f"Fehler: Die Datei {file_path} enthält ungültiges JSON.")
        return {}

def create_default_jobs(num_jobs: int = 12, ops_per_job: int = 3, random_predecessors: bool = True) -> Dict:
    """
    Erstellt eine Standardstruktur mit n Jobs und zufälligen Vorgängern.
    
    Args:
        num_jobs: Anzahl der zu erstellenden Jobs
        ops_per_job: Anzahl der Operationen pro Job (kann variieren)
        random_predecessors: Ob zufällige Vorgänger erstellt werden sollen
        
    Returns:
        Dictionary mit den Standard-Job-Daten
    """
    jobs_data = {"jobs": []}
    machines = ["M1", "M2", "M3", "M4"]
    materials = ["Material_A", "Material_B", "Material_C"]
    hilfsmittel = ["Öl", "Werkzeug", "Schablone", "Kühlmittel"]
    
    for i in range(1, num_jobs + 1):
        # Variiere die Anzahl der Operationen pro Job (2-5)
        actual_ops = random.randint(2, min(5, ops_per_job + 2))
        operations = []
        
        # Erstelle mehrere Operationen pro Job
        for j in range(1, actual_ops + 1):
            # Erste Operation hat keinen Vorgänger
            predecessor = None
            
            # Für Operationen nach der ersten, setze Vorgänger
            if j > 1:
                # Entscheide, ob ein Array von Vorgängern oder ein einzelner Vorgänger
                if random_predecessors:
                    # Standardmäßig ist die vorherige Operation der Vorgänger
                    predecessor = [f"Job_{i}_Op{j-1}"]
                    
                    # Mit geringer Wahrscheinlichkeit füge weitere Vorgänger hinzu
                    if j > 2 and random.random() < 0.2:  # 20% Chance für mehrere Vorgänger
                        additional_pred = random.randint(1, j-2)
                        predecessor.append(f"Job_{i}_Op{additional_pred}")
                else:
                    # Einfacher Vorgänger
                    predecessor = [f"Job_{i}_Op{j-1}"]
            
            # Grundlegende Operation erstellen
            operation = {
                "Name": f"Job_{i}_Op{j}",
                "benötigteZeit": random.randint(20, 60),  # Zufällige Bearbeitungszeit
                "Maschine": random.choice(machines),      # Zufällige Maschine
                "Vorgänger": predecessor,
                "produziertesMaterial": random.choice(materials)
            }
            
            # Zufällig zusätzliche Attribute hinzufügen
            
            # 1. Zwischenlager (30% Chance)
            if random.random() < 0.3:
                operation["zwischenlager"] = {
                    "minVerweildauer": random.randint(10, 20),
                    "lagerkosten": random.randint(1, 5)
                }
            
            # 2. Benötigte Hilfsmittel (40% Chance)
            if random.random() < 0.4:
                # 1-3 zufällige Hilfsmittel
                num_hilfsmittel = random.randint(1, min(3, len(hilfsmittel)))
                selected_hilfsmittel = random.sample(hilfsmittel, num_hilfsmittel)
                operation["benötigteHilfsmittel"] = selected_hilfsmittel
                
                # Wenn Hilfsmittel benötigt werden, füge auch Umrüstkosten hinzu
                operation["umruestkosten"] = random.randint(10, 40)
            
            operations.append(operation)
        
        # Generiere eine zufällige Priorität zwischen 1 und 10
        # Höhere Werte (näher an 10) bedeuten höhere Priorität
        job_priority = random.randint(1, 10)
        
        job = {
            "Name": f"Job_{i}",
            "Priorität": job_priority,
            "Operationen": operations
        }
        jobs_data["jobs"].append(job)
    
    # Umrüstzeiten für verschiedene Maschinen hinzufügen
    jobs_data["maschinenUmrüstzeiten"] = {
        "M1": {"standardZeit": 10, "materialWechsel": 15},
        "M2": {"standardZeit": 15, "materialWechsel": 20},
        "M3": {"standardZeit": 20, "materialWechsel": 25},
        "M4": {"standardZeit": 12, "materialWechsel": 18}
    }
    
    return jobs_data

def load_or_create_jobs(file_path: str, num_jobs: int = 12) -> Dict:
    """
    Lädt eine vorhandene JSON-Datei oder erstellt eine neue mit n Jobs.
    
    Args:
        file_path: Pfad zur JSON-Datei
        num_jobs: Anzahl der zu erstellenden Jobs, falls Datei nicht existiert
        
    Returns:
        Dictionary mit den Job-Daten
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        # Erstelle neue JSON-Datei mit n Jobs
        default_jobs = create_default_jobs(num_jobs)
        create_jobs_json(default_jobs, file_path)
        print(f"Neue Jobs-Datei mit {num_jobs} Jobs erstellt: {file_path}")
        return default_jobs
    except json.JSONDecodeError:
        print(f"Fehler: Die Datei {file_path} enthält ungültiges JSON.")
        return {}

def create_jobs_json(jobs_data: Dict, output_file: str) -> bool:
    """
    Erstellt eine neue JSON-Datei mit den Job-Informationen.
    
    Args:
        jobs_data: Dictionary mit den Job-Daten
        output_file: Pfad für die zu erstellende JSON-Datei
        
    Returns:
        True wenn erfolgreich, False wenn ein Fehler auftritt
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(jobs_data, file, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Fehler beim Erstellen der JSON-Datei: {str(e)}")
        return False

def calculate_setup_time(current_machine: str, previous_material: Optional[str], 
                        next_material: str, setup_data: Dict) -> int:
    """
    Berechnet die Umrüstzeit basierend auf der Maschine und dem Materialwechsel.
    
    Args:
        current_machine: Die aktuelle Maschine
        previous_material: Das vorherige Material (None, wenn es das erste ist)
        next_material: Das nächste zu produzierende Material
        setup_data: Dictionary mit den Umrüstzeit-Informationen
        
    Returns:
        Berechnete Umrüstzeit in Minuten
    """
    # Standard-Umrüstzeit für die Maschine
    if current_machine not in setup_data:
        return 10  # Standard-Fallback, falls Maschine nicht definiert
    
    machine_setup = setup_data.get(current_machine, {})
    standard_time = machine_setup.get("standardZeit", 10)
    
    # Wenn kein vorheriges Material oder gleiches Material, nur Standard-Zeit
    if previous_material is None or previous_material == next_material:
        return standard_time
    
    # Bei Materialwechsel zusätzliche Zeit
    material_change_time = machine_setup.get("materialWechsel", 15)
    return standard_time + material_change_time

def get_job_by_name(jobs_data: Dict, job_name: str) -> Optional[Dict]:
    """
    Findet einen Job anhand seines Namens.
    
    Args:
        jobs_data: Dictionary mit den Job-Daten
        job_name: Name des gesuchten Jobs
        
    Returns:
        Job-Dictionary oder None, wenn nicht gefunden
    """
    for job in jobs_data.get("jobs", []):
        if job.get("Name") == job_name:
            return job
    return None

def get_operation_by_name(job: Dict, operation_name: str) -> Optional[Dict]:
    """
    Findet eine Operation anhand ihres Namens innerhalb eines Jobs.
    
    Args:
        job: Job-Dictionary
        operation_name: Name der gesuchten Operation
        
    Returns:
        Operations-Dictionary oder None, wenn nicht gefunden
    """
    for operation in job.get("Operationen", []):
        if operation.get("Name") == operation_name:
            return operation
    return None

def validate_job_dependencies(jobs_data: Dict) -> List[str]:
    """
    Überprüft, ob alle Vorgänger-Operationen existieren.
    
    Args:
        jobs_data: Dictionary mit den Job-Daten
        
    Returns:
        Liste mit Fehlermeldungen, leere Liste wenn keine Fehler
    """
    errors = []
    
    for job in jobs_data.get("jobs", []):
        job_name = job.get("Name")
        
        for operation in job.get("Operationen", []):
            op_name = operation.get("Name")
            predecessors = operation.get("Vorgänger", [])
            
            # Überspringe, wenn keine Vorgänger
            if not predecessors:
                continue
            
            # Konvertiere einzelnen Vorgänger zu Liste
            if not isinstance(predecessors, list):
                predecessors = [predecessors]
            
            # Prüfe jeden Vorgänger
            for pred in predecessors:
                if pred is None:
                    continue
                    
                # Extrahiere Job-Name aus Operation (Format: "Job_X_OpY")
                parts = pred.split("_Op")
                if len(parts) != 2:
                    errors.append(f"Ungültiges Vorgänger-Format: {pred} für {op_name}")
                    continue
                
                pred_job_name = parts[0]
                pred_op_name = f"{pred_job_name}_Op{parts[1]}"
                
                # Finde den Vorgänger-Job
                pred_job = get_job_by_name(jobs_data, pred_job_name)
                if not pred_job:
                    errors.append(f"Vorgänger-Job nicht gefunden: {pred_job_name} für {op_name}")
                    continue
                
                # Finde die Vorgänger-Operation
                pred_op = get_operation_by_name(pred_job, pred_op_name)
                if not pred_op:
                    errors.append(f"Vorgänger-Operation nicht gefunden: {pred_op_name} für {op_name}")
    
    return errors

# Beispiel für die Verwendung:
if __name__ == "__main__":
    # Pfad zur JSON-Datei
    json_path = os.path.join(os.path.dirname(__file__), "jobs.json")
    
    # JSON-Datei laden oder mit 12 Jobs erstellen, falls nicht vorhanden
    jobs = load_or_create_jobs(json_path, num_jobs=12, ops_per_job=3)
    
    # Beispiel für das Erstellen einer neuen JSON-Datei
    if jobs:
        print(f"Geladene Jobs: {len(jobs.get('jobs', []))}")
        
        # Validiere Job-Abhängigkeiten
        errors = validate_job_dependencies(jobs)
        if errors:
            print("Fehler in den Job-Abhängigkeiten gefunden:")
            for error in errors:
                print(f"  - {error}")
        else:
            print("Alle Job-Abhängigkeiten sind gültig.")
        
        # Beispiel für die Berechnung der Umrüstzeit
        setup_data = jobs.get("maschinenUmrüstzeiten", {})
        setup_time = calculate_setup_time("M1", "Material_A", "Material_B", setup_data)
        print(f"Berechnete Umrüstzeit für M1 (Material A → B): {setup_time} Minuten")
        
        # Erstelle eine Backup-Datei
        backup_path = os.path.join(os.path.dirname(__file__), "jobs_backup.json")
        success = create_jobs_json(jobs, backup_path)
        if success:
            print(f"Backup-Datei wurde erfolgreich erstellt: {backup_path}")