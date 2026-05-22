import numpy as np
import random
import math
import matplotlib.pyplot as plt
import time

# =========================
# DATASET OKUMA
# =========================

with open("TaillardDatasets/tai200x10.txt") as file:
    lines=file.readlines()

# boş satırları temizle
lines= [line.strip() for line in lines if line.strip()]

if not lines:
    raise ValueError("Dosya boş veya okunamadı")
    
# ilk satır: job ve machine sayısı
jobs, machines=map(int, lines[0].split())

# processing time matrix
processing_times = []

for line in lines[1:]:
    row = list(map(int, line.split()))
    processing_times.append(row)

processing_times = np.array(processing_times)

print("Jobs:", jobs)
print("Machines:", machines)

# =========================
# MAKESPAN HESABI
# =========================

def calculate_makespan(sequence, processing_times):

    jobs = len(sequence)
    machines = len(processing_times[0])

    completion = [[0] * machines for _ in range(jobs)]

    for i in range(jobs):
        for j in range(machines):

            current_job = sequence[i]

            # ilk job ilk machine
            if i == 0 and j == 0:

                completion[i][j] = processing_times[current_job][j]

            # ilk job
            elif i == 0:

                completion[i][j] = (
                    completion[i][j - 1]
                    + processing_times[current_job][j]
                )

            # ilk machine
            elif j == 0:

                completion[i][j] = (
                    completion[i - 1][j]
                    + processing_times[current_job][j]
                )

            else:

                completion[i][j] = max(
                    completion[i - 1][j],
                    completion[i][j - 1]
                ) + processing_times[current_job][j]

    return completion[-1][-1]

# =========================
# NEIGHBOR GENERATION
# =========================

def generate_neighbor(sequence):

    neighbor = sequence.copy()

    i, j = random.sample(range(len(sequence)), 2)

    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

    return neighbor

# =========================
# SIMULATED ANNEALING
# =========================

def simulated_annealing(processing_times):

    # başlangıç çözümü
    current_sequence = list(range(len(processing_times)))
    random.shuffle(current_sequence)

    current_makespan = calculate_makespan(
        current_sequence,
        processing_times
    )

    best_sequence = current_sequence.copy()
    best_makespan = current_makespan

    # convergence listesi
    convergence = []

    # SA parametreleri
    temperature = 10000
    cooling_rate = 0.995
    stopping_temperature = 0.001

    while temperature > stopping_temperature:

        # neighbor oluştur
        neighbor_sequence = generate_neighbor(current_sequence)

        neighbor_makespan = calculate_makespan(
            neighbor_sequence,
            processing_times
        )

        # fark
        delta = neighbor_makespan - current_makespan

        # daha iyi çözüm
        if delta < 0:

            current_sequence = neighbor_sequence
            current_makespan = neighbor_makespan

        # daha kötü çözüm bazen kabul edilir
        else:

            probability = math.exp(-delta / temperature)

            if random.random() < probability:

                current_sequence = neighbor_sequence
                current_makespan = neighbor_makespan

        # best çözümü güncelle
        if current_makespan < best_makespan:

            best_sequence = current_sequence.copy()
            best_makespan = current_makespan

        # convergence için current makespan kaydet
        convergence.append(current_makespan)

        # sıcaklığı düşür
        temperature *= cooling_rate

    return best_sequence, best_makespan, convergence

# =========================
# MULTIPLE RUNS
# =========================

results = []

best_overall_convergence = None
best_overall_makespan = float("inf")

start_time = time.time()
for run in range(10):

    best_sequence, best_makespan, convergence = simulated_annealing(
        processing_times
    )

    results.append(best_makespan)

    print(f"\nRun {run+1}")
    print("Best Makespan:", best_makespan)

    # en iyi convergence grafiğini sakla
    if best_makespan < best_overall_makespan:

        best_overall_makespan = best_makespan
        best_overall_convergence = convergence

# =========================
# STATISTICAL RESULTS
# =========================

print("\n========== Statistical Results ==========")

print("Best:", np.min(results))
print("Worst:", np.max(results))
print("Mean:", np.mean(results))
print("Std Dev:", np.std(results))

end_time = time.time()
runtime = end_time - start_time
print("Runtime (seconds):", runtime)

# =========================
# CONVERGENCE GRAPH
# =========================

plt.figure(figsize=(10,5))

plt.plot(best_overall_convergence)

plt.title("Simulated Annealing Convergence Curve")
plt.xlabel("Iteration")
plt.ylabel("Current Makespan")

plt.grid(True)

plt.show()