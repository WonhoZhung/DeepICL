# DeepICL
**Deep** **I**nteraction-**C**onditioned **L**igand generative model


## Train
```
python -u train.py --world_size 4 --save_dir save/exp_0 --data_dir ../data/data/ --key_dir ../data/keys/ --num_layers 5 --num_dense_layers 3 --num_hidden_feature 64 --dist_one_hot_param1 0 15 30 --dist_one_hot_param2 0 10 200 --lr 2e-3 --num_epochs 1001 --save_every 1 --k 8 --vae_loss_beta 0.2 --lr_decay 0.8 --lr_tolerance 4 --lr_min 1e-6 --conditional > log/exp_0.out 2> log/exp_0.err
```

## Sampling
```
python -u generate.py --ngpu 0 --ncpu 16 --k 8 --data_dir ../data/generate_data/ --key_dir ../data/generate_keys/ --restart_dir save/exp_0/save_41.pt --result_dir results/exp_0_0 --num_layers 5 --num_dense_layers 3 --num_hidden_feature 64 --num_sample 100 --max_num_add_atom 30 --dist_one_hot_param1 0 15 30 --dist_one_hot_param2 0 10 200 --temperature_factor1 0.1 --temperature_factor2 0.1 --radial_limits 0.9 2.2 --add_noise --pocket_coeff_max 10.0 --pocket_coeff_thr 2.5 --pocket_coeff_beta 0.91 --conditional --verbose -y
```
