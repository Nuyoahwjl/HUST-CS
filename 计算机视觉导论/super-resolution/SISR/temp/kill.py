import os
import subprocess

def get_gpu_processes():
    try:
        # Use nvidia-smi to get GPU processes
        result = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], encoding='utf-8')
        processes = [line.split(', ') for line in result.strip().split('\n') if line]
        return [(int(pid), name) for pid, name in processes]
    except FileNotFoundError:
        print("nvidia-smi not found. Make sure NVIDIA drivers are installed.")
        return []
    except Exception as e:
        print(f"Error fetching GPU processes: {e}")
        return []

def kill_process(pid):
    try:
        os.kill(pid, 9)  # Send SIGKILL signal
        print(f"Successfully killed process with PID {pid}")
    except Exception as e:
        print(f"Failed to kill process with PID {pid}: {e}")

def main():
    gpu_processes = get_gpu_processes()
    if not gpu_processes:
        print("No GPU processes found.")
        return

    print("GPU processes:")
    for pid, name in gpu_processes:
        print(f"PID: {pid}, Name: {name}")

    for pid, name in gpu_processes:
        confirm = input(f"Do you want to kill the process with PID {pid} (Name: {name})? (yes/no): ").strip().lower()
        if confirm == 'yes':
            kill_process(pid)
        else:
            print(f"Skipped killing process with PID {pid}.")

if __name__ == "__main__":
    main()