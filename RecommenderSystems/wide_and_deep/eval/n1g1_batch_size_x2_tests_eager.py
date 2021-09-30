from eval import Trainer
from config import get_args
import math
from tqdm import tqdm
if __name__ == "__main__":
    batch_size_array=[]
    for i in range(2):
        batch_size_array.append(int(512*math.pow(2,i)))
    for batch_size in tqdm(batch_size_array):
        args=get_args()
        args.execution_mode='eager'
        args.deep_vocab_size=2322444
        args.wide_vocab_size=2322444
        args.hidden_units_num=2
        args.deep_embedding_vec_size=16
        args.batch_size=batch_size
        args.print_interval=100
        args.deep_dropout_rate=0
        args.max_iter=500
        args.eval_name='n1g1_batch_size_x2_tests_eager'
        trainer = Trainer(args)
        trainer()