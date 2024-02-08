# -*- coding:utf-8 -*-
# author: Xinge
# @file: train_cylinder_asym.py

import time
import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data

from utils.load_save_util import load_checkpoint

import warnings

warnings.filterwarnings("ignore")
import wandb


def main(args):
    pytorch_device = torch.device('cuda:0')

    config_path = args.config_path

    configs = load_config_data(config_path)

    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']

    val_batch_size = val_dataloader_config['batch_size']
    train_batch_size = train_dataloader_config['batch_size']

    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']

    model_load_path = train_hypers['model_load_path']
    model_save_path = train_hypers['model_save_path']
    wd = train_hypers['weight_decay']      # weight decay
    amp = train_hypers['mixed_fp16']
    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    my_model = model_builder.build(model_config)
    if os.path.exists(model_load_path):
        my_model = load_checkpoint(model_load_path, my_model)

    my_model.to(pytorch_device)

    # Mixed Precision
    amp_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # EPS is a fixed value to prevent MixedPrecision training errors.   
    optimizer = optim.AdamW(my_model.parameters(), lr=train_hypers["learning_rate"], eps=1e-4, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, train_hypers['max_num_epochs'])

    loss_func, lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
                                                   num_class=num_class, ignore_label=ignore_label)

    train_dataset_loader, val_dataset_loader = data_builder.build(dataset_config,
                                                                  train_dataloader_config,
                                                                  val_dataloader_config,
                                                                  grid_size=grid_size)
    
    wandb.init(project="cylinder3d_dummy_img_fea(sub_data)", name="cylinder3d_lidar+camera-fixed-fus-pos")
    wandb.config = {
        "learning_rate": train_hypers["learning_rate"],
        "epochs": train_hypers["max_num_epochs"],
        "batch_size(train)": train_dataloader_config['batch_size'] 
    }
    
    # training
    epoch = 0
    best_val_miou = 0
    my_model.train()
    global_iter = 0
    check_iter = train_hypers['eval_every_n_steps']

    while epoch < train_hypers['max_num_epochs']:
        #gs
        epoch_start_time = time.time()

        print(f'EPOCH {epoch}')
        loss_list = []
        pbar = tqdm(total=len(train_dataset_loader))
        time.sleep(10)
        # lr_scheduler.step(epoch)

        for i_iter, (_1, train_vox_label, train_grid, _2, train_pt_fea, train_img_fea) in enumerate(train_dataset_loader):
            #gs
            if global_iter % check_iter == 0:
                train_start_time = time.time()

            if global_iter % check_iter == 0 and epoch >= 1:
            #if epoch >= 0: ################################################
                #gs
                val_start_time = time.time()
                
                # Evaluation set
                my_model.eval()
                hist_list = []
                val_loss_list = []
                with torch.no_grad():
                    for i_iter_val, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea, val_img_fea) in enumerate(
                            val_dataset_loader):
                        
                        val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                                          val_pt_fea]
                        val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]
                        val_label_tensor = val_vox_label.type(torch.LongTensor).to(pytorch_device)
                        val_img_fea_ten = [i.type(torch.FloatTensor).to(pytorch_device) for i in val_img_fea]
                                              
                        predict_labels = my_model(val_pt_fea_ten, val_img_fea_ten, val_grid_ten, val_batch_size)
                        # aux_loss = loss_fun(aux_outputs, point_label_tensor)
                        loss = lovasz_softmax(torch.nn.functional.softmax(predict_labels).detach(), val_label_tensor,
                                              ignore=0) + loss_func(predict_labels.detach(), val_label_tensor)
                        predict_labels = torch.argmax(predict_labels, dim=1)
                        predict_labels = predict_labels.cpu().detach().numpy() # [1, 480, 360, 32]
                        # val_grid (num pts, 3)
                        for count, i_val_grid in enumerate(val_grid):
                            hist_list.append(fast_hist_crop(predict_labels[
                                                                count, val_grid[count][:, 0], val_grid[count][:, 1],
                                                                val_grid[count][:, 2]], val_pt_labs[count],
                                                            unique_label))
                        val_loss_list.append(loss.detach().cpu().numpy())
                
                my_model.train()
                iou = per_class_iu(sum(hist_list))
                print('Validation per class iou: ')
                for class_name, class_iou in zip(unique_label_str, iou):
                    print('%s : %.2f%%' % (class_name, class_iou * 100))
                    wandb.log({f'Val_IoU/{class_name}': class_iou * 100}, step=global_iter)
                    
                val_miou = np.nanmean(iou) * 100
                wandb.log({'Val_mIoU': val_miou}, step=global_iter)                
                del val_vox_label, val_grid, val_pt_fea, val_grid_ten

                # save model if performance is improved
                if best_val_miou < val_miou:
                    best_val_miou = val_miou
                    torch.save(my_model.state_dict(), model_save_path)

                print('Current val miou is %.3f while the best val miou is %.3f' %
                      (val_miou, best_val_miou))
                print('Current val loss is %.3f' %
                      (np.mean(val_loss_list)))
                
                #gs
                val_finish_time = time.time()
                print(f'val time {val_finish_time - val_start_time}')

            # Training set
            train_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in train_pt_fea]
            # train_grid_ten = [torch.from_numpy(i[:,:2]).to(pytorch_device) for i in train_grid]
            train_vox_ten = [torch.from_numpy(i).to(pytorch_device) for i in train_grid]
            point_label_tensor = train_vox_label.type(torch.LongTensor).to(pytorch_device)
            train_img_fea_ten = [i.type(torch.FloatTensor).to(pytorch_device) for i in train_img_fea]

            # forward + backward + optimize
            with torch.cuda.amp.autocast(enabled=amp):
                outputs = my_model(train_pt_fea_ten, train_img_fea_ten, train_vox_ten, train_batch_size)
                loss = lovasz_softmax(torch.nn.functional.softmax(outputs), point_label_tensor, ignore=0) + loss_func(
                    outputs, point_label_tensor)
            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()
            # optimizer.step()

            loss_list.append(loss.item())
            wandb.log({'Train_loss': np.mean(loss_list)}, step=global_iter)

            if global_iter % 1000 == 0:
                if len(loss_list) > 0:
                    print('epoch %d iter %5d, loss: %.3f\n' %(epoch, i_iter, np.mean(loss_list)))
                    wandb.log({"Training Loss": {loss.item()}, "Epoch": epoch})
                    
                else:
                    print('loss error')

                # gs
                train_finish_time = time.time()
                print(f'train time {train_finish_time - train_start_time}')

            optimizer.zero_grad()
            pbar.update(1)

            global_iter += 1
            if global_iter % check_iter == 0:
                if len(loss_list) > 0:
                    print('epoch %d iter %5d, loss: %.3f\n' %(epoch, i_iter, np.mean(loss_list)))
                else:
                    print('loss error')
            
        scheduler.step()    # Updates cosine annealing
        pbar.close()
        epoch += 1

        #gs
        epoch_finish_time = time.time()
        print(f'EPOCH TIME {epoch_finish_time - epoch_start_time}')



if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/semantickitti.yaml')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
