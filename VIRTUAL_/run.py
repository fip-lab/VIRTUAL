import subprocess
import time

def run_one_shot(top_m, top_n):
    cmd = f'python one_shot.py --top_m {top_m} --top_n {top_n}'
    process = subprocess.Popen(cmd.split())
    process.wait()
    time.sleep(5)  

def main():
    # 参数组合,你可以添加更多参数进行网格搜索
    params = [
        (1, 3)
    ]
    
    for top_m, top_n in params:
        run_one_shot(top_m, top_n)

if __name__ == "__main__":
    main()