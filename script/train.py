import os
import time
import sys
import numpy as np
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch

import arguments
from model import DeepSLIP
from dataset import PDBbindDataset
import utils
import traceback
from datetime import datetime

import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def sample_to_device(sample, device):
    sample["whole"].to(device)
    sample["traj"].to(device)
    return

def train(model, args, optimizer, data, train, device=None, scaler=None):
    model.train() if train else model.eval()

    i_batch = 0
    total_losses, vae_losses, type_losses, dist_losses, ssl_losses = \
            [], [], [], [], []
    while True:
        sample = next(data, None)
        if sample is None:
            break

        sample_to_device(sample, device)
        
        if args.autocast:
            with amp.autocast():
                try:
                    total_loss, vae_loss, type_loss, dist_loss, ssl_loss = model(sample)
                except Exception as e:
                    print(traceback.format_exc())
                    exit()
        else:
            try:
                total_loss, vae_loss, type_loss, dist_loss, ssl_loss = model(sample)
            except Exception as e:
                print(traceback.format_exc())
                exit()
        if train:
            optimizer.zero_grad()
            if args.autocast:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

        vae_losses.append(vae_loss.data.cpu().numpy())
        type_losses.append(type_loss.data.cpu().numpy())
        dist_losses.append(dist_loss.data.cpu().numpy())
        total_losses.append(total_loss.data.cpu().numpy())
        if args.ssl:
            ssl_losses.append(ssl_loss.data.cpu().numpy())

    vae_losses = np.mean(np.array(vae_losses))
    type_losses = np.mean(np.array(type_losses))
    dist_losses = np.mean(np.array(dist_losses))
    total_losses = np.mean(np.array(total_losses))
    if args.ssl:
        ssl_losses = np.mean(np.array(ssl_losses))
        return total_losses, vae_losses, type_losses, dist_losses, ssl_losses

    return total_losses, vae_losses, type_losses, dist_losses, 0.0


def main_worker(gpu, ngpus_per_node, args):

    ############ Distributed Data Parallel #############
    # https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization
    rank = gpu
    #print("Rank:", rank, flush=True)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = args.master_port

    dist.init_process_group(
            "nccl", 
            rank=rank, 
            world_size=args.world_size
    )
    print(utils.text_filling(f"Finished Setting DDP: CUDA:{rank}"), flush=True)
    ####################################################

    # Path
    save_dir = utils.get_abs_path(args.save_dir)
    data_dir = utils.get_abs_path(args.data_dir)
    if args.restart_file:
        restart_file = utils.get_abs_path(args.restart_file)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Dataloader
    train_dataset = PDBbindDataset(args, mode='train')
    valid_dataset = PDBbindDataset(args, mode='valid')

    ############ Distributed Data Parallel #############
    train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=args.world_size,
            rank=rank
    )
    valid_sampler = DistributedSampler(
            valid_dataset,
            num_replicas=args.world_size,
            rank=rank,
            shuffle=False
    )
    ####################################################
    train_dataloader = DataLoader(
            train_dataset,
            1,
            num_workers=0,
            pin_memory=True,
            sampler=train_sampler
    )
    valid_dataloader = DataLoader(
            valid_dataset,
            1, 
            num_workers=0,
            pin_memory=True,
            shuffle=False,
            sampler=valid_sampler
    )
    N_TRAIN_DATA = len(train_dataset)
    N_VALID_DATA = len(valid_dataset)
    if not args.restart_file and rank == 0:
        print("Train dataset length: ", N_TRAIN_DATA, flush=True)
        print("Validation dataset length: ", N_VALID_DATA, flush=True)
        print(utils.text_filling("Finished Loading Datasets"), flush=True)

    # Model initialize
    model = DeepSLIP(args)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    model = utils.initialize_model(model, args.world_size > 0, args.restart_file)

    ############ Distributed Data Parallel #############
    # Wrap the model
    model = DDP(model, device_ids=[gpu], find_unused_parameters=True)
    #model = DDP(model, device_ids=[gpu])
    cudnn.benchmark = True
    ####################################################

    if not args.restart_file and rank == 0:
        print("Number of Parameters: ", \
              sum(p.numel() for p in model.parameters() if p.requires_grad), \
              flush=True)
        print(utils.text_filling("Finished Loading Model"), flush=True)

    # Optimizer
    optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=args.lr,
            weight_decay=args.weight_decay
    )

    # Scaler (AMP)
    if args.autocast:
        scaler = amp.GradScaler()
    else:
        scaler = None

    # VAE loss annealing
    vae_coeff_init, vae_coeff_final = args.vae_loss_coeff

    # Train
    if args.restart_file:
        start_epoch = int(args.restart_file.split("_")[-1].split(".")[0])
    else:
        start_epoch = 0

    best_epoch = 0
    min_valid_loss = 1e6
    lr_tick = 0
    for epoch in range(start_epoch, args.num_epochs):
        if epoch == 0 and rank == 0:
            print(f"EPOCH || " + \
                  f"TRA_VAE | " + \
                  f"TRA_SSL | " + \
                  f"TRA_TYPE | " + \
                  f"TRA_DIST | " + \
                  f"TRA_TOT || " + \
                  f"VAL_VAE | " + \
                  f"VAL_SSL | " + \
                  f"VAL_TYPE | " + \
                  f"VAL_DIST | " + \
                  f"VAL_TOT || " + \
                  f"TIME/EPOCH | " + \
                  f"LR | " + \
                  f"BEST_EPOCH", flush=True)

        train_data = iter(train_dataloader)
        valid_data = iter(valid_dataloader)
        
        # KL annealing
        args.vae_coeff = vae_coeff_final + \
                (vae_coeff_init - vae_coeff_final) * \
                ((1 - args.vae_loss_beta)**epoch)

        st = time.time()

        train_total_losses, train_vae_losses, train_type_losses, \
                train_dist_losses, train_ssl_losses = \
                train(
                        model=model, 
                        args=args,
                        optimizer=optimizer,
                        data=train_data,
                        train=True,
                        device=gpu,
                        scaler=scaler
                )
        
        # validation process
        valid_total_losses, valid_vae_losses, valid_type_losses, \
                valid_dist_losses, valid_ssl_losses = \
                train(
                        model=model, 
                        args=args,
                        optimizer=optimizer,
                        data=valid_data,
                        train=False,
                        device=gpu,
                        scaler=scaler
                )

        et = time.time()
        
        if valid_total_losses < min_valid_loss:
            min_valid_loss = valid_total_losses
            best_epoch = epoch
            lr_tick = 0
        else:
            lr_tick += 1

        if lr_tick >= args.lr_tolerance:
            for param_group in optimizer.param_groups:
                lr = param_group["lr"]
                if lr > args.lr_min:
                    param_group["lr"] = lr * args.lr_decay
        
        if rank == 0:
            print(f"{epoch} || " + \
                  f"{train_vae_losses:.3f} | " + \
                  f"{train_ssl_losses:.3f} | " + \
                  f"{train_type_losses:.3f} | " + \
                  f"{train_dist_losses:.3f} | " + \
                  f"{train_total_losses:.3f} || " + \
                  f"{valid_vae_losses:.3f} | " + \
                  f"{valid_ssl_losses:.3f} | " + \
                  f"{valid_type_losses:.3f} | " + \
                  f"{valid_dist_losses:.3f} | " + \
                  f"{valid_total_losses:.3f} || " + \
                  f"{(et - st):.2f} | " + \
                  f"{[group['lr'] for group in optimizer.param_groups][0]:.4f} | " + \
                  f"{best_epoch}{'*' if lr_tick==0 else ''}", flush=True)


        # Save model
        name = os.path.join(save_dir, f"save_{epoch}.pt")
        save_every = 1 if not args.save_every else args.save_every
        if epoch % save_every == 0 and rank == 0:
            torch.save(model.state_dict(), name)



def main():

    now = datetime.now()
    print(f"Train starts at {now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}-{now.second}...")

    args = arguments.train_args_parser()
    d = vars(args)
    print(utils.text_filling("PARAMETER SETTINGS"), flush=True)
    for a in d: print(a, "=", d[a])
    print(80*'#', flush=True)

    args.master_port = utils.find_free_port()

    args.distributed = args.world_size > 1
    os.environ['CUDA_VISIBLE_DEVICES'] = utils.get_cuda_visible_devices(args.world_size)
    if args.distributed:
        mp.spawn(
                main_worker, 
                nprocs=args.world_size, 
                args=(args.world_size, args,),
        )
    else:
        main_worker(0, args.world_size, args)


if __name__ == "__main__":
    
    main()

