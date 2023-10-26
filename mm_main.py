import torch
from tqdm import tqdm
import argparse
import numpy as np
from util.pc_rotate import PointcloudRotate

# from Dataloader.model_net_cross_val import get_sets


# from Dataloader.shapenet_cross_val import get_sets # not used
# from Dataloader.modelnet40_fs import get_sets # not used


from util.get_acc import cal_cfm
import torch.nn as nn

# ======== load model =========
# from model.network import fs_network
# from model.network_gait import fs_network
from model.mm_network import fs_network
import logging

import os
from torch.utils.tensorboard import SummaryWriter
import json
import yaml

# ============== Get Configuration =================
def get_arg():
    cfg=argparse.ArgumentParser()
    cfg.add_argument('--exp_name',default='exp1')
    cfg.add_argument('--solo_base_train', action='store_true', help='train the sla merely on base classes')
    cfg.add_argument('--exp_des', default=' exp1:')
    cfg.add_argument('--multigpu',default=False)
    cfg.add_argument('--epochs',default=130)
    cfg.add_argument('--decay_ep',default=5)
    cfg.add_argument('--gamma',default=0.7)
    cfg.add_argument('--lr',default=0.0001)
    cfg.add_argument('--train',default=True)
    cfg.add_argument('--seed',default=0)
    cfg.add_argument('--device',default='cuda')
    cfg.add_argument('--lr_sch',default=False)
    cfg.add_argument('--weight_decay', default=0.000001, type=float)
    cfg.add_argument('--data_aug',default=True)
    cfg.add_argument('--val_epoch_size', type=int, default=700)



    # ======== few shot cfg =============#
    cfg.add_argument('--k_way',default=5, type=int)
    cfg.add_argument('--n_shot',default=1, type=int)
    cfg.add_argument('--query',default=8, type=int)
    cfg.add_argument('--backbone',default='dgcnn_mm4',choices=['dgcnn'])
    cfg.add_argument('--fs_head',type=str,default='protonet',choices=['protonet','cia'])
    cfg.add_argument('--fold',default=0, type=int)
    # ===================================#


    # ======== path needed ==============#
    cfg.add_argument('--project_path',default='./')
    
    cfg.add_argument('--data_path',default='')
    cfg.add_argument('--dataset', default='modelnet40', choices=['shapenet', 'modelnet40', 'modelnet40c', 'scanobjectnn'])
    cfg.add_argument('--base_classes', default=10, choices=[30, 10])
    cfg.add_argument('--exp_folder_name',default='ModelNet40_cross')
    # ===================================#
    
    return cfg.parse_args()
cfg=get_arg()
# ==================================================
# SETUP PATH FOR SAVING EXPERIMENTAL RESULTS 
# exp_path=os.path.join(cfg.project_path,cfg.exp_folder_name,cfg.exp_name)
# if not os.path.exists(exp_path):
#     os.makedirs(exp_path)
        
# SETUP DATASET PATH AND NUMBER OF BASE CLASSES
if cfg.dataset == 'scanobjectnn':
    from Dataloader.scanobjectnn_cross_val import get_sets
    cfg.data_path = '/data1/minmin/scanobjectnn/ScanObjectNN_fs_cross_validation/Data'
    cfg.base_classes = 10
elif cfg.dataset == 'modelnet40':
    from Dataloader.model_net_cross_val import get_sets
    cfg.data_path = '/data1/minmin/modelnet40_fs_crossvalidation'
    cfg.base_classes = 30
elif cfg.dataset == 'modelnet40c':
    from Dataloader.model_net_cross_val import get_sets
    cfg.data_path = '/data1/minmin/ModelNet40_C_fewshot'
    cfg.base_classes = 30
else:
    raise Exception("dataset error")

# ============= create logging ==============
def get_logger(file_name='accuracy.log'):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s, %(name)s, %(message)s')

    ########### this is used to set the log file ##########
    file_handler = logging.FileHandler(file_name)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    #######################################################


    ######### this is used to set the output in the terminal/screen ########
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    #################################################################

    ####### add the log file handler and terminal handerler to the logger #######
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    ##############################################################################

    return logger
# ============================================

def test_model(model,val_loader,cfg):
    global logger
    logger=get_logger(file_name=os.path.join(cfg.exp_name, 'testing_result.log'))

    exp_path = os.path.join(cfg.exp_name,'pth_file')
    picked_pth = sorted(os.listdir(exp_path),key=lambda x:int(x.split('_')[-1]))[-1]
    pth_file = torch.load(os.path.join(exp_path,picked_pth))
    model.load_state_dict(pth_file['model_state'])
    

    model=model.cuda()
    bar=tqdm(val_loader,ncols=100,unit='batch',leave=False)
    summary=run_one_epoch(model,bar,'test',loss_func=None)

    acc_list=summary['acc']

    mean_acc=np.mean(acc_list)
    std_acc=np.std(acc_list)

    interval=1.960*(std_acc/np.sqrt(len(acc_list)))
    logger.debug('Mean: {}, Interval: {}'.format(mean_acc,interval))





def main(cfg):
    global logger
    # logger=get_logger()


    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled=False
    
    # IF SOLO_BASE_TRAIN=TRUE, SET QUERY=0

    train_loader,val_loader = get_sets(data_path=cfg.data_path, fold=cfg.fold, k_way=cfg.k_way, n_shot=cfg.n_shot, query_num=cfg.query)

    model = fs_network(cfg)
    
    if cfg.multigpu:
        model=nn.DataParallel(model)
    
    if cfg.train:
        train_model(model,train_loader,val_loader,cfg)
    
    else:
        test_model(model,val_loader,cfg)
    


def train_model(model,train_loader,val_loader,cfg):
    device=torch.device(cfg.device)
    model=model.to(device)
    
    #====== loss and optimizer =======
    loss_func=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg.lr)
    if cfg.lr_sch:
        lr_schedule=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=np.arange(10,cfg.epochs,cfg.decay_ep),gamma=cfg.gamma)
    
    
    def train_one_epoch():
        bar=tqdm(train_loader,ncols=100,unit='batch',leave=False)
        epsum=run_one_epoch(model,bar,'train',loss_func=loss_func,optimizer=optimizer)
        summary={"loss/train":np.mean(epsum['loss'])}
        # summary['train_sppt_acc'] = np.mean(epsum['train_sppt_acc'])
        return summary
        
        
    def eval_one_epoch():
        bar=tqdm(val_loader,ncols=100,unit='batch',leave=False)
        epsum=run_one_epoch(model, bar,"valid",loss_func=loss_func)
        mean_acc=np.mean(epsum['acc'])
        summary={'meac/valid':mean_acc}
        test_accuracies = np.array(epsum['acc'])
        test_accuracies = np.reshape(test_accuracies, -1)
        stds = np.std(test_accuracies, 0)
        ci95 = 1.96 * stds / np.sqrt(cfg.val_epoch_size)
        summary['std/valid'] = ci95
        summary["loss/valid"]=np.mean(epsum['loss'])
        return summary,epsum['cfm']
    
    
    # ======== define exp path ===========
    exp_path=cfg.exp_name
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    logger=get_logger(os.path.join(exp_path, 'accuracy.log'))
    # save config into json #
    cfg_dict=vars(cfg)
    yaml_file=os.path.join(exp_path,'config.yaml')
    with open(yaml_file,'w') as outfile:
        yaml.dump(cfg_dict, outfile, default_flow_style=False)
    # f = open(json_file, "w")
    # json.dump(cfg_dict, f)
    # f.close()
    #########################
    
    tensorboard=SummaryWriter(log_dir=os.path.join(exp_path,'TB'),purge_step=cfg.epochs)
    pth_path=os.path.join(exp_path,'pth_file')
    if not os.path.exists(pth_path):
        os.mkdir(pth_path)
    # =====================================
    
    # ========= train start ===============
    acc_list=[]
    sppt_acc_list = []
    tqdm_epochs=tqdm(range(cfg.epochs),unit='epoch',ncols=100)
    for e in tqdm_epochs:
        train_summary=train_one_epoch()
        val_summary,conf_mat=eval_one_epoch()
        summary={**train_summary,**val_summary}
        
        if cfg.lr_sch:
            lr_schedule.step()
        
        # accuracy=val_summary['meac']
        accuracy=val_summary['meac/valid']
        std = val_summary['std/valid']
        # sppt_acc = train_summary['train_sppt_acc']
        # sppt_acc_list.append(train_summary['train_sppt_acc'])
        acc_list.append(val_summary['meac/valid'])
        
        logger.debug('Epoch {}: . FSL Acc: {:.5f}, std: {:.2%}. Highest FSL Acc: {:.5f}'.format(e, accuracy, std, np.max(acc_list)))
        # logger.debug('Epoch {}: . Sppt Acc: {:.5f}. Highest Sppt Acc: {:.5f}'.format(e, sppt_acc, np.max(sppt_acc_list)))
        # print('epoch {}: {}. Highese: {}'.format(e,accuracy,np.max(acc_list)))
        
        if np.max(acc_list)==acc_list[-1]:
            summary_saved={**summary,
                            'model_state':model.state_dict(),
                            'optimizer_state':optimizer.state_dict(),
                            'cfm':conf_mat}
            torch.save(summary_saved,os.path.join(pth_path,'best.pth'))
        
        for name,val in summary.items():
            tensorboard.add_scalar(name,val,e)
    
    # =======================================    
    
    



def run_one_epoch(model,bar,mode,loss_func,optimizer=None,show_interval=10):
    confusion_mat=np.zeros((cfg.k_way,cfg.k_way))
    summary={"acc":[],"loss":[]}
    device=next(model.parameters()).device
    
    if mode=='train':
        model.train()
    else:
        model.eval()
    
    for i, (x_cpu,y_cpu) in enumerate(bar):
        x, y = x_cpu.to(device), y_cpu.to(device)
        sppt_y = y[:cfg.k_way*cfg.n_shot]
        
        if mode=='train':
            q_pred, loss = model(x, mixup_hidden=True)
            
            #==take one step==#
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #=================#
        else:
            with torch.no_grad():
                q_pred, loss = model(x, mixup_hidden=False)
        
        
        summary['loss']+=[loss.item()]
        
        if mode=='train':
            if i%show_interval==0:
                bar.set_description("Loss: %.3f"%(np.mean(summary['loss'])))
            # sppt_ssl_acc = sum(torch.argmax(s_pred, dim=-1) == ssl_sppt_target)*1.0 / ssl_sppt_target.size(0)
            # summary['train_sppt_acc'] += [sppt_ssl_acc.item()]
                
        else:
            batch_cfm=cal_cfm(q_pred, model.q_label, ncls=cfg.k_way)
            batch_acc=np.trace(batch_cfm)/np.sum(batch_cfm)
            summary['acc'].append(batch_acc)
            if i%show_interval==0:
                bar.set_description("mea_ac: %.3f"%(np.mean(summary['acc'])))
            
            confusion_mat+=batch_cfm
    
    if mode!='train':
        summary['cfm']=confusion_mat
    
    return summary
            



if __name__=='__main__':
    main(cfg)
    