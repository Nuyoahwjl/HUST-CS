import threading  
import time        
import os        

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

def read_input(file):
    for line in file:
        line = line.strip()    
        if line:              
            yield line.split(', ') 
    return file

def mapper(readfile, writefile):
    os.makedirs(os.path.dirname(writefile), exist_ok=True)
    with open(readfile, 'r') as file:
        lines = read_input(file) 
        with open(writefile, 'w') as f:
            for words in lines:  
                for word in words: 
                    f.write(f"{word},1\n")


if __name__ == '__main__':
    output_dir = os.path.join(project_root, 'output/1-map-res')
    os.makedirs(output_dir, exist_ok=True)
   
    threads = [] 
    for i in range(1, 10):
        source_file = os.path.join(project_root, 'data', f'source{i:02d}')
        output_file = os.path.join(output_dir, f'map{i}')
        thread = threading.Thread(target=mapper, args=(source_file, output_file))
        threads.append(thread)
        thread.start()  # 启动线程

    # 记录程序开始时间
    start = time.perf_counter()
    
    print("===map===")
    # 等待所有线程完成并计算耗时
    for idx, thread in enumerate(threads, 1):
        thread.join()
        elapsed = time.perf_counter() - start 
        print(f"t{idx}: {elapsed:.6f} s") 

    print("All threads completed.")