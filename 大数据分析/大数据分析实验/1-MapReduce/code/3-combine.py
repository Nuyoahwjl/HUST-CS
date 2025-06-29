import os
import time
import threading

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

def combiner(input_path, output_path):
    word_counts = {}
    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(',')
                if len(parts) == 2:
                    word = parts[0].strip()
                    count = int(parts[1].strip())
                    word_counts[word] = word_counts.get(word, 0) + count
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        for word, total in sorted(word_counts.items()):
            f.write(f"{word},{total}\n")


if __name__ == '__main__':
    map_res_dir = os.path.join(project_root, 'output/1-map-res')
    combine_dir = os.path.join(project_root, 'output/3-combine-res')
    
    os.makedirs(combine_dir, exist_ok=True)
    
    threads = []
    for i in range(1, 10):
        input_file = os.path.join(map_res_dir, f'map{i}')
        output_file = os.path.join(combine_dir, f'combine{i}')
        thread = threading.Thread(target=combiner, args=(input_file, output_file))
        threads.append(thread)
        thread.start()
    
    start = time.perf_counter()
    
    print("===combine===")
    for idx, thread in enumerate(threads, 1):
        thread.join()
        elapsed = time.perf_counter() - start 
        print(f"t{idx}: {elapsed:.6f} s") 
    
    print("All threads completed.")