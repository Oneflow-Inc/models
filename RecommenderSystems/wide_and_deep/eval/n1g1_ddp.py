from eval_new import Trainer
from config import get_args
import math
from tqdm import tqdm
if __name__ == "__main__":
    args=get_args()
    trainer = Trainer(args)
    trainer()