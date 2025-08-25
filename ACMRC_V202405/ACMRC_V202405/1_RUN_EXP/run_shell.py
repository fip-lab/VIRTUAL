import subprocess
import re
import os
import sys, pty
import datetime
from utils.excel_utils import get_latest_excel, read_excel, write_result,write_main_table

# 2>&1 | tee /home/huanggj/out.txt

def task_filter(task_list):
    run_task = []
    for task_info in task_list:
        # 不运行的任务
        if task_info['is_run'] == 'no':
            continue
        # 已经运行完的任务
        if task_info['is_finished'] == '√':
            continue
        run_task.append(task_info)
    return run_task


def shell_format(params):
    # 提取并从字典中移除 main_class
    main_class = params.pop('main_class', None)
    command = 'python -u '
    if main_class is not None:
        command += f' {main_class}'
    for key, value in params.items():
        command += f' --{key} {value}'
    shell = f'source {conda_path} && conda activate {envs_path} && export PYTHONPATH={project_path} && {command}'  # + ' 2>&1 | tee /home/huanggj/out.txt'
    return shell

def write_log(log_str):
    time_ = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    with open('../log/' + time_ +  '_log.txt', 'a+') as f:
        f.write(log_str)

# def execute_command(cmd):
#     process = subprocess.Popen(cmd, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT,bufsize=1)
#
#     # Poll process for new output until finished
#     while True:
#         nextline = process.stdout.readline()
#         if not nextline and process.poll() is not None:
#             break
#         sys.stdout.write(nextline.decode('utf-8'))
#         sys.stdout.flush()
#
#     output = process.communicate()[0]
#     exitCode = process.returncode
#
#     if (exitCode == 0):
#         return output
#     else:
#         raise subprocess.CalledProcessError(exitCode, cmd)

def execute_command(cmd, log_file):
    master, slave = pty.openpty()
    process = subprocess.Popen(cmd, shell=True, executable='/bin/bash', stdout=slave, stderr=subprocess.STDOUT)
    os.close(slave)

    output_string = ""
    with open(log_file, 'w') as f:
        while True:
            try:
                output = os.read(master, 1024).decode()
                if not output:
                    break
                f.write(output)
                sys.stdout.write(output)
                sys.stdout.flush()
                output_string += output
            except OSError:
                break

    exitCode = process.wait()

    if (exitCode == 0):
        return output_string
    else:
        raise subprocess.CalledProcessError(exitCode, cmd)



# conda_path = '/disk2/anaconda3/etc/profile.d/conda.sh'
# envs_path = '/disk2/anaconda3/envs/acmrc_huanggj'
# project_path = "/disk2/huanggj/ACMRC_EXP_V202306"

conda_path = '/home/zhourc/anaconda3/etc/profile.d/conda.sh'
envs_path = '/home/zhourc/anaconda3/envs/acmrc_huanggj'
project_path = "/home/zhourc/ACMRC_V202405/ACMRC_V202405/1_RUN_EXP"


if __name__ == '__main__':

    # 是否要回填任务到大表中
    fill_sheet = False
    # # 如果不设置此目标excel文件，则去寻找指定目录中修改时间最新的excel文件
    # excel_file = '/disk2/huanggj/ACMRC_EXP_V202306/4_RESULT/exp_20240314.xlsx'
    # # 输出文件的目录
    # target_dir = '/disk2/huanggj/ACMRC_EXP_V202306/4_RESULT/'
    # # ckpt文件输出目录
    # checkpoint_dir = '/disk2/huanggj/ACMRC_EXP_V202306/output/exp_20240302'
    # # 日志输出目录
    # log_dir = '/disk2/huanggj/ACMRC_EXP_V202306/log/'
    excel_file = './4_RESULT/exp_20240314.xlsx'
    target_dir = './4_RESULT/'
    checkpoint_dir = './output/exp_20240302'
    log_dir = './log/'


    if excel_file == '':
        excel_file = get_latest_excel(target_dir)

    print("任务读取中........")
    task_start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 创建log文件夹保存模型的checkpoint
    checkpoint_dir = os.path.join(checkpoint_dir, excel_file.split('/')[-1].split('.')[0])
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # 创建程序所有日志的文件夹
    log_dir = os.path.join(log_dir, excel_file.split('/')[-1].split('.')[0])
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 1. 读取excel表格，获取任务列表
    column_type_dict = {'task_id': int, 'epochs': int, 'batch_size': int}
    task_list = read_excel(excel_file, column_type_dict)
    # 2. 过滤不需要运行的任务
    task_list = task_filter(task_list)

    curr_task_cnt = 0
    res_list = []
    print(f"实验总数: {len(task_list)}")
    # 3. 遍历shell命令列表，执行，记录执行结果到excel
    for task in task_list:
        print(f"#########  本次实验任务id - {task.get('task_id')}  ########")
        # checkpoint输出目录
        task['output_dir'] = checkpoint_dir
        # 格式化shell
        shell = shell_format(task)
        # 记录开始时间
        start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')



        # 程序运行
        print(shell)
        print_ = ''
        try:
            print_ = execute_command(shell, log_dir + '/' +  str(task.get('task_id')) + '_' + datetime.datetime.now().strftime('%m%d%H%M') +  '.log')
        except subprocess.CalledProcessError as e:
            print(f"Shell command '{shell}' failed with error message: \n{str(e)}")

        # 记录结束时间
        end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 获取test的结果
        # test_acc_ = ''
        # if task['is_10_fold'] == 'yes':
        #     test_acc_ = re.findall(r"Avg accuracy across 10-folds: (.+?)%", print_)
        # else:
        #     test_acc_ = re.findall(r"Test Acc: (.+?)%",print_)

        test_acc_ = re.findall(r"Test Acc: (.+?)%",print_)

        if len(test_acc_) == 0 :
            raise Exception("无法获取test acc")
        result = float(test_acc_[0])
        task['result'] = result
        res_list.append(result)

        # 写入数据
        data_dict = {"task_id":task.get("task_id"), "start_time":start_time, "end_time":end_time, "result":result}
        write_result(excel_file,data_dict, column_type_dict)

        print(f"本次任务id - {task.get('task_id')} , 总体完成度 : {curr_task_cnt}/{len(task_list)}")
        curr_task_cnt += 1

    task_end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print("=========      所有实验任务结束      ===========")
    print(f"开始时间: {task_start_time}")
    print(f"结束时间: {task_end_time}")
    print(f"实验总数: {len(task_list)}")
    print(f"最好结果: {max(res_list)}")
    print(f"最差结果: {min(res_list)}")
    print("=========   别看了，先去南村爽吃一把   ===========")

    # 4. 填表  任务表->实验大表  默认任务表序号为0，实验大表序号为1
    if(fill_sheet):
        write_main_table(excel_file, task_list)

