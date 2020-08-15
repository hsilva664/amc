import os

models = ['vgg', 'resnet']
seeds = [1234, 2234, 3234]
wrem_list = [0.5, 0.25, 0.14, 0.07, 0.03, 0.02, 0.01, 0.005]
l_bound_list = [0.2, 0.1, 0.5, 0.02, 0.01, 0.006, 0.004, 0.0002]

for seed in seeds:
    for wrem, l_bound in zip(wrem_list, l_bound_list):
        for model in models:
            in_dir = os.path.join('checkpoints', model, str(seed), 'training_state.tar' )
            out_dir = os.path.join('checkpoints', model, str(seed), 'pruned_{wrem}.tar'.format(wrem=wrem) )
            print("python amc_search.py --model={model} --dataset=cifar10 --data_root=./dataset/cifar10 --preserve_ratio={wrem} --lbound={l_bound} --rbound=1 --reward=acc_reward --ckpt_path={in_dir} --seed={seed} --n_worker=2 --export_path={export_path}".format(model=model, wrem=wrem, in_dir=in_dir, seed=seed, export_path=out_dir, l_bound=l_bound))
