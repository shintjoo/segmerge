import matplotlib.pyplot as plt
import numpy as np

def parse_results(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    cpu_times = []
    gpu_sorting_times = []
    gpu_no_sorting_times = []

    current_section = None
    for line in lines:
        line = line.strip()
        
        if line.startswith('# CPU'):
            current_section = 'CPU'
            continue
        elif line.startswith('# GPU with sorting'):
            current_section = 'GPU_SORTING'
            continue
        elif line.startswith('# GPU without sorting'):
            current_section = 'GPU_NO_SORTING'
            continue
        elif line.startswith('#') or not line:
            continue

        try:
            value = int(line)
            if current_section == 'CPU':
                cpu_times.append(value)
            elif current_section == 'GPU_SORTING':
                gpu_sorting_times.append(value)
            elif current_section == 'GPU_NO_SORTING':
                gpu_no_sorting_times.append(value)
        except ValueError:
            print(f"Skipping invalid line: {line}")

    return cpu_times, gpu_sorting_times, gpu_no_sorting_times

def plot_results(cpu_times, gpu_sorting_times, gpu_no_sorting_times):
    n_values = [2**m for m in range(8, 29)]

    plt.figure(figsize=(12, 8))

    plt.plot(n_values, cpu_times, marker='o', label='CPU')
    plt.plot(n_values, gpu_sorting_times, marker='o', label='GPU with sorting')
    plt.plot(n_values, gpu_no_sorting_times, marker='o', label='GPU without sorting')

    plt.xscale('log', base=2)
    plt.yscale('log')

    plt.xlabel('n (size of input arrays A and B, n = 2^m)')
    plt.ylabel('Runtime (microseconds)')
    plt.title('SegMerge Runtime Comparison (CPU vs GPU)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()

    plt.show()

if __name__ == '__main__':
    file_path = 'results.txt'
    cpu_times, gpu_sorting_times, gpu_no_sorting_times = parse_results(file_path)
    plot_results(cpu_times, gpu_sorting_times, gpu_no_sorting_times)
