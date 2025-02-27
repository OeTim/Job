# ob Scheduling Environment Dokumentation
## Überblick
Die JobSchedulingEnv -Klasse implementiert eine Reinforcement-Learning-Umgebung für die Optimierung von Job-Scheduling unter Verwendung der OpenAI Gym-Schnittstelle. Sie simuliert eine Produktionsumgebung, in der Jobs mit mehreren Operationen auf verschiedenen Maschinen geplant werden müssen, um den Makespan (Gesamtfertigstellungszeit) zu minimieren.

## Umgebungsbeschreibung
Diese Umgebung modelliert ein Job-Shop-Scheduling-Problem, bei dem:

- Jeder Job aus mehreren Operationen besteht
- Operationen Abhängigkeiten (Vorgänger) haben
- Maschinen nur eine Operation gleichzeitig verarbeiten können
- Das Ziel die Minimierung des Makespans ist
## Klasse: JobSchedulingEnv
### Initialisierung
def __init__(self, data, config=None)
- Zweck : Erstellt eine Job-Scheduling-Umgebung aus Produktionsdaten
- Parameter :
  - data : Dictionary mit Jobs und deren Operationen
  - config : Optionale Konfigurationsparameter
### Hauptmethoden reset()
- Zweck : Setzt die Umgebung auf ihren Ausgangszustand zurück
- Rückgabe : Initialer Beobachtungszustand
- Details : Initialisiert Jobs, Maschinen und Zeitverfolgungsvariablen step(action)
- Zweck : Führt eine Scheduling-Entscheidung aus
- Parameter :
  - action : Job-Index für die Planung oder Name der Scheduling-Strategie
- Rückgabe : (Beobachtung, Belohnung, Fertig, Info)
- Details : Plant den ausgewählten Job und berechnet die Belohnung basierend auf Makespan, Fertigstellungszeit und Maschinenauslastung render(mode='human')
- Zweck : Zeigt den aktuellen Zustand der Umgebung an
- Details : Zeigt aktuelle Zeit, Makespan und Maschinenverfügbarkeit
### Hilfsmethoden _execute_job(job)
- Zweck : Simuliert die Ausführung der Operationen eines Jobs
- Rückgabe : Fertigstellungszeit des Jobs
- Details : Behandelt Operationssequenzierung, Maschinenzuweisung und Rüstzeiten _get_ready_operations()
- Zweck : Identifiziert Operationen, die zur Ausführung bereit sind
- Details : Prüft Operationsabhängigkeiten, um zu bestimmen, welche geplant werden können _get_observation()
- Zweck : Erstellt den Beobachtungsvektor für den aktuellen Zustand
- Rückgabe : Dictionary mit normalisierten Features für Jobs, Maschinen und Zeit
- Details : Normalisiert verschiedene Job- und Maschinenmerkmale für den RL-Agenten
### Scheduling-Strategien _fifo_schedule()
- Zweck : First In, First Out Scheduling-Strategie
- Details : Wählt den ersten Job in der Liste der verbleibenden Jobs _lifo_schedule()
- Zweck : Last In, First Out Scheduling-Strategie
- Details : Wählt den letzten Job in der Liste der verbleibenden Jobs _spt_schedule()
- Zweck : Shortest Processing Time Scheduling-Strategie
- Details : Wählt den Job mit der kürzesten Gesamtbearbeitungszeit
## Beobachtungsraum
Die Umgebung bietet einen strukturierten Beobachtungsraum mit drei Komponenten:

1. waiting_jobs : Merkmale der zu verarbeitenden Jobs (10 Merkmale pro Job)
2. machine_status : Status jeder Maschine (3 Merkmale pro Maschine)
3. time_features : Globale zeitbezogene Merkmale (3 Merkmale)
## Aktionsraum
Der Aktionsraum ist diskret und repräsentiert den Index des als nächstes zu planenden Jobs.

## Belohnungsfunktion
Die Belohnungsfunktion ist eine gewichtete Kombination aus:

- Makespan-Differenz (60%): Bestraft Erhöhungen des Makespans
- Fertigstellungszeit (30%): Fördert schnelles Abschließen von Jobs
- Maschinenauslastung (10%): Belohnt effiziente Maschinennutzung

        # Umgebung erstellen
        env = JobSchedulingEnv(produktionsdaten)

        # Umgebung zurücksetzen
        obs = env.reset()

        # Eine Episode ausführen
        done = False
        while not done:
            # Aktion auswählen (zu planender Job)
            action = agent.choose_action(obs)
            
            # Aktion ausführen
            obs, reward, done, info = env.step(action)
            
            # Aktuellen Zustand anzeigen
            env.render()

        # Finalen Makespan ausgeben
        print(f"Finaler Makespan: {info['makespan']}")