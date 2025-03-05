import random
import json
import os

def generate_synthetic_data(config):
    """Generiert synthetische Produktionsdaten"""
    jobs = []

    for i in range(1, config["n_jobs"] + 1):
        n_ops = random.randint(config["min_ops"], config["max_ops"])
        operations = []

        for j in range(1, n_ops + 1):
            operation = {
                "Name": f"Job_{i}_Op{j}",
                "benötigteZeit": random.randint(20, 60),
                "Maschine": random.choice(config["machines"]),
                "Vorgänger": [f"Job_{i}_Op{j-1}"] if j > 1 else None,
                "produziertesMaterial": random.choice(config["materials"])
            }

            # Zufällig zusätzliche Attribute hinzufügen
            if random.random() < 0.4:  # 40% Wahrscheinlichkeit für Hilfsmittel
                n_tools = random.randint(1, 2)
                operation["benötigteHilfsmittel"] = random.sample(config["tools"], n_tools)
                # operation["umruestzeit"] = random.randint(1, 10)
                operation["umruestkosten"] = random.randint(10, 50)

            if random.random() < 0.3:  # 30% Wahrscheinlichkeit für Zwischenlager
                operation["zwischenlager"] = {
                    "minVerweildauer": random.randint(5, 20),
                    "lagerkosten": random.randint(1, 5)
                }

            operations.append(operation)

        job = {
            "Name": f"Job_{i}",
            "Priorität": random.randint(1, 10),
            "Operationen": operations
        }

        jobs.append(job)

    generated_data = {"jobs": jobs}
    print(f"✅ Synthetische Daten für {config['n_jobs']} Jobs generiert")
    return generated_data


def save_data(data, filename="production_data.json"):
    """Speichert die generierten Daten in eine JSON-Datei"""
    filepath = os.path.join(os.getcwd(), filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"✅ Daten gespeichert in: {filepath}")
    return filepath


def load_data(filename="production_data.json"):
    """Lädt Daten aus einer JSON-Datei"""
    filepath = os.path.join(os.getcwd(), filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✅ Daten geladen aus: {filepath}")
    return data
