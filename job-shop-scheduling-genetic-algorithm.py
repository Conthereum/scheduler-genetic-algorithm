import random
import numpy as np
import time
from typing import List, Tuple

# Constants
NUM_PROCESSES = 200  # Renamed from NUM_TASKS to NUM_PROCESSES
# NUM_PROCESSES = 4
NUM_COMPUTERS = 3  # Renamed from NUM_SHOPS to NUM_COMPUTERS
PROCESS_DURATION_MIN = 5
PROCESS_DURATION_MAX = 10
CONFLICT_PROBABILITY = 0.45
POPULATION_SIZE = 100
GENERATIONS = 500
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.8
RANDOM_SEED = 42

# Set random seed
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Generate process durations
process_durations = [random.randint(PROCESS_DURATION_MIN, PROCESS_DURATION_MAX) for _ in range(NUM_PROCESSES)]

# Generate process conflict matrix with symmetry
conflict_matrix = np.random.rand(NUM_PROCESSES, NUM_PROCESSES) < CONFLICT_PROBABILITY
np.fill_diagonal(conflict_matrix, False)  # No process conflicts with itself
# Ensure the matrix is symmetric (if i conflicts with j, j also conflicts with i)
conflict_matrix = np.triu(conflict_matrix, 1)  # Keep only the upper triangle of the matrix
conflict_matrix = conflict_matrix + conflict_matrix.T  # Make the matrix symmetric by adding the transpose


# def generate_valid_initial_schedule() -> List[Tuple[int, int, int]]:
#     """Generate a valid and optimized initial schedule using a heuristic approach."""
#     schedule = []
#     computer_end_times = [0] * NUM_COMPUTERS  # Track the end time for each computer
#
#     # Sort processes by their duration, from longest to shortest
#     sorted_processes = sorted(range(NUM_PROCESSES), key=lambda p: -process_durations[p])
#
#     # Iterate over the sorted processes and assign them to computers
#     for process in sorted_processes:
#         assigned = False
#
#         # Try to assign the process to a computer in a way that minimizes conflicts
#         for computer in range(NUM_COMPUTERS):
#             # Check for conflicts: Ensure no conflicting processes are assigned to the same computer
#             conflicting_processes = [other_process for other_process in schedule if conflict_matrix[process, other_process[0]]]
#
#             # Check if the computer is available (no conflict or overlap)
#             if all(
#                     start_time + process_durations[process] <= computer_end_times[computer] or
#                     other_computer != computer
#                     for _, other_computer, start_time in conflicting_processes
#             ):
#                 # Assign the process to this computer
#                 start_time = max(computer_end_times[computer], 0)  # No earlier than the computer's last task end time
#                 schedule.append((process, computer, start_time))
#                 computer_end_times[computer] = start_time + process_durations[process]  # Update the end time for this computer
#                 assigned = True
#                 break
#
#         # If not assigned to any computer yet, fallback to brute force and assign
#         if not assigned:
#             for computer in range(NUM_COMPUTERS):
#                 start_time = max(computer_end_times[computer], 0)
#                 schedule.append((process, computer, start_time))
#                 computer_end_times[computer] = start_time + process_durations[process]
#
#     return schedule

import random
from typing import List, Tuple

def generate_valid_initial_population(population_size: int) -> List[List[Tuple[int, int, int]]]:
    """Generate a valid and diverse initial population using a heuristic approach."""
    population = []

    for _ in range(population_size):
        schedule = []  # A single schedule for one solution
        computer_end_times = [0] * NUM_COMPUTERS  # Track the end time for each computer

        # Sort processes by their duration, from longest to shortest
        sorted_processes = sorted(range(NUM_PROCESSES), key=lambda p: -process_durations[p])

        # Iterate over the sorted processes and assign them to computers
        for process in sorted_processes:
            assigned = False

            # Try to assign the process to a computer in a way that minimizes conflicts
            available_computers = list(range(NUM_COMPUTERS))  # All available computers

            # Shuffle the available computers to introduce randomness and avoid bias
            random.shuffle(available_computers)

            for computer in available_computers:
                # Check for conflicts: Ensure no conflicting processes are assigned to the same computer
                conflicting_processes = [other_process for other_process in schedule if conflict_matrix[process, other_process[0]]]

                # Check if the computer is available (no conflict or overlap)
                if all(
                        start_time + process_durations[process] <= computer_end_times[computer] or
                        other_computer != computer
                        for _, other_computer, start_time in conflicting_processes
                ):
                    # Assign the process to this computer
                    start_time = max(computer_end_times[computer], 0)  # No earlier than the computer's last task end time
                    schedule.append((process, computer, start_time))
                    computer_end_times[computer] = start_time + process_durations[process]  # Update the end time for this computer
                    assigned = True
                    break

            # If not assigned to any computer yet, fallback to brute force and assign
            if not assigned:
                # Pick a computer randomly (this is a fallback option if no valid assignment found)
                computer = random.choice(range(NUM_COMPUTERS))
                start_time = max(computer_end_times[computer], 0)
                schedule.append((process, computer, start_time))
                computer_end_times[computer] = start_time + process_durations[process]

        population.append(schedule)

    return population



# def generate_schedule() -> List[Tuple[int, int, int]]:
#     """Generate a random schedule (chromosome) assigning processes to computers and start times."""
#     schedule = []
#     computer_end_times = [0] * NUM_COMPUTERS
#     for process in range(NUM_PROCESSES):
#         computer = random.randint(0, NUM_COMPUTERS - 1)
#         start_time = computer_end_times[computer]
#         schedule.append((process, computer, start_time))
#         computer_end_times[computer] = start_time + process_durations[process]
#     return schedule


def calculate_completion_time(schedule: List[Tuple[int, int, int]]) -> int:
    """Calculate the completion time for the given schedule."""
    computer_end_times = [0] * NUM_COMPUTERS
    for process, computer, start_time in schedule:
        finish_time = start_time + process_durations[process]
        computer_end_times[computer] = max(computer_end_times[computer], finish_time)
    return max(computer_end_times)


def calculate_fitness(schedule: List[Tuple[int, int, int]]) -> int:
    """Calculate fitness as the inverse of the completion time."""
    return -calculate_completion_time(schedule)  # Minimize completion time


def repair(schedule: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    """Validate and repair schedules to avoid conflicts efficiently."""
    computer_end_times = [0] * NUM_COMPUTERS
    computer_processes = {i: [] for i in range(NUM_COMPUTERS)}  # Track processes scheduled on each computer
    repaired_schedule = []

    for process, computer, start_time in schedule:
        # Conflict detection: Only check the processes already scheduled on the same computer
        conflicting_processes = computer_processes[computer]

        # Check if the start_time conflicts with any process already scheduled on the same computer
        conflicting = False
        for other_process, other_start_time in conflicting_processes:
            if start_time < other_start_time + process_durations[other_process] and start_time + process_durations[process] > other_start_time:
                conflicting = True
                break

        if conflicting:
            # Resolve conflict by moving start_time to the earliest available time slot
            start_time = max(computer_end_times[computer], start_time)

            # Jump forward in time until there are no conflicts
            while any(start_time < other_start_time + process_durations[other_process] and start_time + process_durations[process] > other_start_time for other_process, other_start_time in conflicting_processes):
                start_time += 1  # Increment the start_time

        # Add the process with its adjusted start time
        repaired_schedule.append((process, computer, start_time))
        computer_end_times[computer] = start_time + process_durations[process]  # Update end time for the computer
        computer_processes[computer].append((process, start_time))  # Add process to the computer's scheduled processes

    return repaired_schedule



def select_parents(population: List[List[Tuple[int, int, int]]], fitnesses: List[int]) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    """Select two parents using fitness-proportional selection."""
    total_fitness = sum(fitnesses)
    probabilities = [1 - (f / total_fitness) for f in fitnesses]
    probabilities = np.cumsum(np.array(probabilities) / sum(probabilities))
    indices = [np.searchsorted(probabilities, np.random.rand()) for _ in range(2)]
    return [population[indices[0]], population[indices[1]]]


def crossover(parent1: List[Tuple[int, int, int]], parent2: List[Tuple[int, int, int]]) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    """Perform single-point crossover between two parents."""
    if random.random() > CROSSOVER_RATE:
        return parent1, parent2
    point = random.randint(1, NUM_PROCESSES - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def mutate(schedule: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    """Mutate a schedule."""
    if random.random() < MUTATION_RATE:
        index = random.randint(0, NUM_PROCESSES - 1)
        process, _, _ = schedule[index]
        new_computer = random.randint(0, NUM_COMPUTERS - 1)
        new_start_time = random.randint(0, sum(process_durations))  # Use horizon
        schedule[index] = (process, new_computer, new_start_time)
    return schedule

# def validate_final_schedule(schedule: List[Tuple[int, int, int]]) -> str:
#     """Validate the final schedule to ensure no conflicts or overlapping processes on the same computer."""
#     # Track the end time for each computer to check for overlaps
#     computer_end_times = {i: [] for i in range(NUM_COMPUTERS)}
#
#     # Iterate through the schedule
#     for process, computer, start_time in schedule:
#         finish_time = start_time + process_durations[process]
#
#         # Check if this process conflicts with another process on the same computer
#         for other_process, other_computer, other_start_time in computer_end_times[computer]:
#             other_finish_time = other_start_time + process_durations[other_process]
#             # If the process overlaps in time with another process, return "Invalid"
#             if start_time < other_finish_time and finish_time > other_start_time:
#                 return "Invalid"  # Overlapping processes on the same computer
#
#         # Check if there are any conflicting processes (if any) from the conflict matrix
#         for other_process, other_computer, other_start_time in schedule:
#             if process != other_process and computer == other_computer:
#                 if conflict_matrix[process, other_process]:
#                     return "Invalid"  # Conflicting processes running at the same time
#
#         # Add the process's finish time to the end times list for this computer
#         computer_end_times[computer].append((process, start_time))
#
#     return "Valid"
def validate_final_schedule(schedule: List[Tuple[int, int, int]]) -> str:
    """Validate the final schedule."""
    # Check for conflicts and overlapping processes on the same computer
    computer_end_times = {i: [] for i in range(NUM_COMPUTERS)}  # Dictionary to track process end times per computer

    for process, computer, start_time in schedule:
        finish_time = start_time + process_durations[process]

        # Check for conflicts with other processes assigned to the same computer
        for other_process, other_computer, other_start_time in schedule:
            if process != other_process and computer == other_computer:
                # Check if there is any overlap in time
                other_finish_time = other_start_time + process_durations[other_process]
                if (start_time < other_finish_time and finish_time > other_start_time):
                    return "Invalid"  # Overlapping processes on the same computer

                # Check for conflicts between this process and the other process
                if conflict_matrix[process, other_process]:
                    return "Invalid"  # Conflicting processes running at the same time

        computer_end_times[computer].append(finish_time)

    return "Valid"


def genetic_algorithm() -> Tuple[List[Tuple[int, int, int]], int, float]:
    """Run the genetic algorithm."""
    start_time = time.time()
    population = generate_valid_initial_population(POPULATION_SIZE)
        # [generate_valid_initial_schedule() for _ in range(POPULATION_SIZE)]

    print(f"\nInitial population (startTime:processId:duration):")
    for sch in population:
        print_schedule(sch)
        print("\n")

    for generation in range(GENERATIONS):
        fitnesses = [calculate_fitness(schedule) for schedule in population]
        new_population = []

        for _ in range(POPULATION_SIZE // 2):
            parent1, parent2 = select_parents(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(repair(mutate(child1)))
            new_population.append(repair(mutate(child2)))

        population = new_population

    fitnesses = [calculate_fitness(schedule) for schedule in population]
    best_index = np.argmax(fitnesses)
    best_schedule = population[best_index]
    optimal_schedule_time = -fitnesses[best_index]
    wall_time = (time.time() - start_time) * 1000  # Convert to milliseconds

    return best_schedule, optimal_schedule_time, wall_time

def print_initial_information():
    """Print the initial information about the processes, conflict matrix, and other relevant data."""
    print("Initial Process Durations:")
    for i, duration in enumerate(process_durations):
        print(f"Process {i}: {duration} ms")

    print("\nInitial Conflict Matrix:")
    for i in range(NUM_PROCESSES):
        for j in range(NUM_PROCESSES):
            print(f"{int(conflict_matrix[i, j])} ", end=" ")
        print()  # New line after each row

    # print(f"\nTotal number of processes: {NUM_PROCESSES}")
    # print(f"Total number of computers: {NUM_COMPUTERS}")
    # print(f"Process Duration Range: {PROCESS_DURATION_MIN} ms to {PROCESS_DURATION_MAX} ms")
    # print(f"Conflict Probability: {CONFLICT_PROBABILITY}")
    # print(f"Population Size for Genetic Algorithm: {POPULATION_SIZE}")
    # print(f"Number of Generations for Genetic Algorithm: {GENERATIONS}")
    # print(f"Mutation Rate: {MUTATION_RATE}")
    # print(f"Crossover Rate: {CROSSOVER_RATE}")


def print_schedule(schedule_to_print: List[Tuple[int, int, int]]):
    """Print the schedule in a more visual format for each computer."""
    # Create a dictionary to group processes by computer
    computers_schedule = {i: [] for i in range(NUM_COMPUTERS)}

    # Group the processes by computer
    for process_id, computer_id, start_time in schedule_to_print:
        duration = process_durations[process_id]
        computers_schedule[computer_id].append((start_time, process_id, duration))

    # Sort processes by start time for each computer (optional but makes it easier to read)
    for computer_id in computers_schedule:
        computers_schedule[computer_id].sort()  # Sort by start time

    # Print the schedule for each computer in the desired format
    for computer_id in range(NUM_COMPUTERS):
        # Start the line with the computer ID
        schedule_line = f"c{computer_id} "

        # Append each process (start_time:pX:duration) to the line
        schedule_line += " ".join([f"{start_time}:p{process_id}:{duration}"
                                   for start_time, process_id, duration in computers_schedule[computer_id]])

        # Print the schedule line for this computer
        print(schedule_line)


if __name__ == "__main__":
    print_initial_information()

    serial_time_horizon = sum(process_durations)

    best_schedule, optimal_schedule_time, wall_time = genetic_algorithm()

    speedup = serial_time_horizon / (wall_time + optimal_schedule_time)

    # Validate the final schedule
    validation_result = validate_final_schedule(best_schedule)



    # print(f"\n\nBest Schedule (Process ID, Computer ID, Start Time): {best_schedule}")
    # Print the best schedule in the visual format
    print(f"\nBest Schedule (startTime:processId:duration):")
    print_schedule(best_schedule)

    print(f"\nOptimal Schedule Time: {optimal_schedule_time} ms")
    print(f"Wall Time: {wall_time:.2f} ms")
    print(f"Serial Time Horizon: {serial_time_horizon} ms")
    print(f"Speedup: {speedup:.2f}")
    print(f"Final Validation: {validation_result}")
