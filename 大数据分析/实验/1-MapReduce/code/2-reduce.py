import os
import time   
import threading
from collections import defaultdict

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

def reducer(input_files, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    word_counts = defaultdict(int)
    for file_path in input_files:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        word, count = line.rsplit(',', 1) 
                        word_counts[word] += int(count)
                    except ValueError:
                        continue

    sorted_items = sorted(word_counts.items(), key=lambda x: x[0])
    
    with open(output_file, 'w') as f:
        for word, total in sorted_items:
            f.write(f"{word},{total}\n")


if __name__ == '__main__':
    map_res_dir = os.path.join(project_root, 'output/1-map-res')
    reduce_res_dir = os.path.join(project_root, 'output/2-reduce-res')
    os.makedirs(reduce_res_dir, exist_ok=True)

    threads = []
    for reducer_id in range(1, 4):
        start_map = (reducer_id - 1) * 3 + 1
        input_files = [
            os.path.join(map_res_dir, f'map{num}')
            for num in range(start_map, start_map + 3)
        ]
        
        output_file = os.path.join(reduce_res_dir, f'reduce{reducer_id}')
        
        thread = threading.Thread(
            target=reducer,
            args=(input_files, output_file)
        )
        threads.append(thread)
        thread.start()

    start = time.perf_counter()
    
    print("===reduce===")
    for idx, thread in enumerate(threads, 1):
        thread.join()
        elapsed = time.perf_counter() - start 
        print(f"t{idx}: {elapsed:.6f} s") 
        
    print("All threads completed.")