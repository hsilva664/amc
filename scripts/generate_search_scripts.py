import os

models = ['vgg', 'resnet']
seeds = [1234, 2234, 3234]
wrem_list = [0.5, 0.25, 0.14, 0.07, 0.03, 0.02, 0.01, 0.005]

for seed in seeds:
    for wrem in wrem_list:
        for model in models:
            in_dir = os.path.join('checkpoints', model, str(seed), 'training_state.tar' )
            out_dir = os.path.join('checkpoints', model, str(seed), 'pruned_{wrem}.tar'.format(wrem=wrem) )
            print("python3 amc_search.py --model={model} --dataset=cifar10 --data_root=./dataset/cifar10 --preserve_ratio={wrem} --lbound=0.2 --rbound=1 --reward=acc_reward --ckpt_path={in_dir} --seed={seed} --n_worker=2 --export_path={export_path}".format(model=model, wrem=wrem, in_dir=in_dir, seed=seed, export_path=out_dir))
