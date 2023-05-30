import logging
from pathlib import Path

from ms_level.train import train as train_mulit_scale
from helpers import init_helper, data_helper
from multiprocessing import Pool
import os, time, random
logger = logging.getLogger()

TRAINER = {
    'ms': train_mulit_scale
}


def get_trainer(model_type):
    assert model_type in TRAINER
    return TRAINER[model_type]

def test22(name):
    print("print name:{}".format(name))
    return 100

def test(name,tt):
    print('Run task %s %s (%s)...' % (name,tt, os.getpid()))
    start = time.time()
    f = test22(name)
    time.sleep(random.random() * 3)
    end = time.time()
    print('Task %s runs %0.2f seconds.' % (name, (end - start)))

def trainMultiProcess(args,split,model_dir,split_path,split_idx,results,trainer):
    print('Run task (%s)...' % (os.getpid()))
    logger.info(f'Start training on {split_path.stem}: split {split_idx}')
    ckpt_path = data_helper.get_ckpt_path(model_dir, split_path, split_idx)
    fscore = trainer(args, split, ckpt_path, split_idx=split_idx)
    # stats.update(fscore=fscore)
    results[f'split{split_idx}'] = float(fscore)

def main():
    args = init_helper.get_arguments()

    init_helper.init_logger(args.model_dir, args.log_file)
    init_helper.set_random_seed(args.seed)

    logger.info(vars(args))

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    data_helper.get_ckpt_dir(model_dir).mkdir(parents=True, exist_ok=True)

    trainer = get_trainer(args.model)

    data_helper.dump_yaml(vars(args), model_dir / 'args.yml')

    for split_path in args.splits:
        split_path = Path(split_path)
        splits = data_helper.load_yaml(split_path)

        results = {}
        #stats = data_helper.AverageMeter('fscore')
        # i = 0
        training_pool = Pool(5)
        for split_idx, split in enumerate(splits):
            # i += 1
            # if i == args.split_num:
            #trainMultiProcess(args,split,model_dir,split_path,split_idx,results,trainer)
            #training_pool.apply_async(test, args=(split_idx,split_idx,))
            training_pool.apply_async(trainMultiProcess, (args,split,model_dir,split_path,split_idx,results,trainer,))
        print('Waiting for all subprocesses done...')
        training_pool.close()
        training_pool.join()
        print('All subprocesses done.')
        # i = 0
        # sum_fscore = 0.0
        # for value in results.values():
        #     sum_fscore += value
        #     i += 1
        # results['mean'] = sum_fscore/i
        # data_helper.dump_yaml(results, model_dir / f'{split_path.stem}.yml')

        # logger.info(f'Training done on {split_path.stem}. F-score: {stats.fscore:.4f}')


if __name__ == '__main__':
    main()
