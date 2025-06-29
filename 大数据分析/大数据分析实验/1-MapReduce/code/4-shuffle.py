import os
import threading
import hashlib
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

def get_reducer_id(word, num_reducers=3):
    hash_bytes = hashlib.sha256(word.encode()).digest()  # 使用SHA-256哈希
    hash_int = int.from_bytes(hash_bytes[:4], byteorder='big')  # 取前4字节转为整数
    return hash_int % num_reducers

def shuffle(input_file, output_dir, locks):
    # 初始化三个Reducer的缓冲区
    buffers = {0: [], 1: [], 2: []}
    # 读取输入文件并分组
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                word, count = line.split(',', 1)
                word = word.strip()
            except ValueError:
                continue
            reducer_id = get_reducer_id(word)
            buffers[reducer_id].append(f"{word},{count}\n")
    
    for reducer_id in (0, 1, 2):
        if not buffers[reducer_id]:
            continue
        with locks[reducer_id - 1]:
            output_path = os.path.join(output_dir, f"shuffle{reducer_id+1}")
            with open(output_path, 'a') as f:
                f.writelines(buffers[reducer_id])


if __name__ == '__main__':
    # 初始化路径
    combine_dir = os.path.join(project_root, 'output/3-combine-res')
    shuffle_dir = os.path.join(project_root, 'output/4-shuffle-res')
    
    # 确保输出目录存在
    os.makedirs(shuffle_dir, exist_ok=True)
    
    locks = [threading.Lock() for _ in range(3)]
    
    for i in range(3):
        output_file = os.path.join(shuffle_dir, f"shuffle{i+1}")
        if not os.path.exists(output_file):
            open(output_file, 'w').close()
    
    # 创建并启动线程
    threads = []
    for i in range(1, 10):
        combine_file = os.path.join(combine_dir, f"combine{i}")
        thread = threading.Thread(
            target=shuffle,
            args=(combine_file, shuffle_dir, locks)
        )
        threads.append(thread)
        thread.start()
    
    start = time.perf_counter()
    
    print("===shuffle===")
    for idx, thread in enumerate(threads, 1):
        thread.join()
        elapsed = time.perf_counter() - start 
        print(f"t{idx}: {elapsed:.6f} s") 
    
    print("All threads completed.")