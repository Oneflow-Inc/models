import argparse
import os
import glob
from statistics import median




def write_line(f, lst, separator=',', start_end=False):
    lst = ['', *lst, ''] if start_end else lst
    f.write(separator.join(lst))
    f.write('\n')


def value_format(value):
    if isinstance(value, float):
        return '{:.3f}'.format(value)
    elif isinstance(value, int):
        return f'{value:,}'
    else:
        return str(value)


def extract_mem_info(mem_file):
    if not os.path.isfile(mem_file):
        return 'NA'

    with open(mem_file, 'r') as f:
        for line in f.readlines():
            ss = line.split(' ')
            if len(ss) < 5:
                continue
            if ss[0] == 'max':
                return int(float(ss[-1].strip()) / 1024 /1024)
    return 'NA'

 
def extract_info_from_file(log_file):
    '''
    ------------------------ arguments ------------------------
      batch_size ...................................... 256
      ctrl_port ....................................... 50051
      data_dir ........................................ /dataset/wdl_ofrecord/ofrecord
      dataset_format .................................. ofrecord
      ddp ............................................. False
      deep_dropout_rate ............................... 0.5
      deep_embedding_vec_size ......................... 16
      deep_vocab_size ................................. 2322444
      eval_batchs ..................................... 0
      execution_mode .................................. graph
      gpu_num_per_node ................................ 1
      hf_deep_vocab_size .............................. 800000
      hf_wide_vocab_size .............................. 800000
      hidden_size ..................................... 1024
      hidden_units_num ................................ 2
      learning_rate ................................... 0.001
      max_iter ........................................ 1100
      model_load_dir ..................................
      model_save_dir ..................................
      node_ips ........................................ ['127.0.0.1']
      num_dataloader_thread_per_gpu ................... 2
      num_deep_sparse_fields .......................... 26
      num_dense_fields ................................ 13
      num_nodes ....................................... 1
      num_wide_sparse_fields .......................... 2
      print_interval .................................. 100
      save_initial_model .............................. False
      use_single_dataloader_thread .................... False
      wide_vocab_size ................................. 2322444
    -------------------- end of arguments ---------------------
    '''
    # extract info from file name
    result_dict = {}
    with open(log_file, 'r') as f:
        first_iter = 0
        first_time = 0
        for line in f.readlines():
            ss = line.strip().split(' ')
            if ss[0] in ['num_nodes', 'gpu_num_per_node', 'batch_size', 'deep_vocab_size','hidden_units_num', 'deep_embedding_vec_size']:
                result_dict[ss[0]] = ss[-1].strip() 
            elif len(ss) > 3 and ss[2] == 'iter':
                if first_iter == 0:
                    first_iter = int(ss[3].strip())
                    first_time = float(ss[-1].strip())
                    result_dict['latency(ms)'] = 0
                elif int(ss[3].strip()) - first_iter == 1000:
                    result_dict['latency(ms)'] = (float(ss[-1].strip()) - first_time) 
    mem = extract_mem_info(log_file[:-3] + 'mem')
    result_dict['memory_usage(MB)'] = mem
    return result_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="flags for OneFlow wide&deep")
    parser.add_argument("--benchmark_log_dir", type=str, required=True)
    args = parser.parse_args()

    logs_list = glob.glob(os.path.join(args.benchmark_log_dir, "*.log"))
    #logs_list = sorted(logs_list)
    chunk_list = {}
    for log_file in logs_list:
        test_result = extract_info_from_file(log_file)
        print(test_result)
        #json_file = os.path.basename(log_file)[:-4]
        json_file = os.path.basename(log_file)[:-13]
        print(json_file)
        test_result['log_file'] = json_file
        if json_file not in chunk_list.keys():
            chunk_list[json_file] = []
        chunk_list[json_file].append(test_result)
    result_list = []
    for log_name,chunk in chunk_list.items():
        latency_list = []
        for single_result in chunk:
            latency_list.append(single_result['latency(ms)'])
        tmp_chunk = chunk[0]
        tmp_chunk['gpu'] = 'n{}g{}'.format(tmp_chunk['num_nodes'], tmp_chunk['gpu_num_per_node'])
        tmp_chunk['latency(ms)'] = median(latency_list)
        result_list.append(tmp_chunk)
    #with open(os.path.join(args.benchmark_log_dir, 'latency_reprot.md'), 'w') as f:
    report_file = args.benchmark_log_dir + '_latency_report.md'
    with open(report_file, 'w') as f:
        titles = ['log_file', 'gpu', 'batch_size', 'deep_vocab_size','deep_embedding_vec_size', 'hidden_units_num', 'latency(ms)', 'memory_usage(MB)']
        write_line(f, titles, '|', True)
        write_line(f, ['----' for _ in titles], '|', True)
        for result in result_list:
            if 'latency(ms)' not in result.keys():
                print(result['log_file'], 'is not complete!')
                continue
            cells = [value_format(result[title]) for title in titles]
            write_line(f, cells, '|', True)
