import streamlit as st
import numpy as np
import random
import matplotlib.pyplot as plt

# ======================
# DATASET
# ======================
def generate_jobs(num_jobs):
    return np.random.randint(1, 10, size=num_jobs)

# ======================
# EVALUATION
# ======================
def evaluate(schedule, jobs, num_machines):
    machine_time = [0] * num_machines
    waiting_time = 0

    for job in schedule:
        machine = machine_time.index(min(machine_time))
        waiting_time += machine_time[machine]
        machine_time[machine] += jobs[job]

    makespan = max(machine_time)
    idle_time = sum(makespan - t for t in machine_time)

    return makespan, waiting_time, idle_time

def fitness(schedule, jobs, machines):
    m, w, i = evaluate(schedule, jobs, machines)
    return 0.5*m + 0.3*w + 0.2*i

# ======================
# GENETIC ALGORITHM
# ======================
def genetic_algorithm(jobs, machines, pop_size, generations, mutation_rate):
    num_jobs = len(jobs)
    population = [random.sample(range(num_jobs), num_jobs) for _ in range(pop_size)]
    history = []

    for _ in range(generations):
        population.sort(key=lambda s: fitness(s, jobs, machines))
        history.append(fitness(population[0], jobs, machines))

        next_gen = population[:2]

        while len(next_gen) < pop_size:
            p1, p2 = random.sample(population[:10], 2)
            cut = random.randint(1, num_jobs - 1)
            child = p1[:cut] + [j for j in p2 if j not in p1[:cut]]

            if random.random() < mutation_rate:
                a, b = random.sample(range(num_jobs), 2)
                child[a], child[b] = child[b], child[a]

            next_gen.append(child)

        population = next_gen

    return population[0], history

# ======================
# GANTT CHART
# ======================
def plot_gantt(schedule, jobs, num_machines):
    machine_time = [0] * num_machines
    gantt = []

    for job in schedule:
        machine = machine_time.index(min(machine_time))
        start = machine_time[machine]
        duration = jobs[job]
        gantt.append((machine, start, duration, job))
        machine_time[machine] += duration

    fig, ax = plt.subplots()
    for machine, start, duration, job in gantt:
        ax.barh(machine, duration, left=start)
        ax.text(start + duration/2, machine, f"J{job+1}",
                ha='center', va='center', color='white')

    ax.set_yticks(range(num_machines))
    ax.set_yticklabels([f"Machine {i+1}" for i in range(num_machines)])
    ax.set_xlabel("Time")
    ax.set_title("Gantt Chart")
    return fig

# ======================
# STREAMLIT UI
# ======================
st.title("Multi-Objective Job Scheduling using Genetic Algorithm")

num_jobs = st.slider("Number of Jobs", 5, 15, 8)
num_machines = st.slider("Number of Machines", 1, 5, 3)
pop_size = st.slider("Population Size", 10, 100, 30)
generations = st.slider("Generations", 10, 200, 50)
mutation_rate = st.slider("Mutation Rate", 0.0, 0.5, 0.1)

jobs = generate_jobs(num_jobs)

# ======================
# TABLE (INI YANG AWAK NAK)
# ======================
st.subheader("Job Dataset (Dummy Data)")
st.table({
    "Job ID": [f"J{i+1}" for i in range(num_jobs)],
    "Processing Time": jobs
})

# ======================
# RUN BUTTON
# ======================
if st.button("Run Genetic Algorithm"):
    best, history = genetic_algorithm(
        jobs, num_machines, pop_size, generations, mutation_rate
    )

    m, w, i = evaluate(best, jobs, num_machines)

    st.subheader("Optimisation Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Makespan", m)
    col2.metric("Waiting Time", w)
    col3.metric("Idle Time", i)

    st.subheader("Fitness Convergence")
    fig1, ax1 = plt.subplots()
    ax1.plot(history)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness Value")
    st.pyplot(fig1)

    st.subheader("Final Schedule")
    st.pyplot(plot_gantt(best, jobs, num_machines))
