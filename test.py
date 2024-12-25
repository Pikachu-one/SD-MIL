import time
import torch
import wandb
import numpy as np
import torch.nn as nn
from dataloader import *
from torch.utils.data import DataLoader, RandomSampler
import argparse, os
from modules import attmil,clam,dsmil,transmil,mean_max,rrt,rrt_origin,attmil_ibmil
import torch.backends.cudnn as cudnn
from torch.nn.functional import one_hot
from modules import attmil_prototype_self_learning

from torch.cuda.amp import GradScaler
from contextlib import suppress
import time

from timm.utils import AverageMeter,dispatch_clip_grad
from timm.models import  model_parameters
from collections import OrderedDict

from utils import *
from torch.utils.data.distributed import DistributedSampler

def main(args):
    # set seed
    seed_torch(args.seed)
    # --->generate dataset
    if args.datasets.lower() == 'tcga':
        label_path=os.path.join(args.dataset_root,'label.csv')
        p, l = get_patient_label(label_path)
        index = [i for i in range(len(p))]
        random.shuffle(index)
        p = p[index]
        l = l[index]

    if args.cv_fold > 1:
        train_p, train_l, test_p, test_l,val_p,val_l = get_kflod(args.cv_fold, p, l,args.val_ratio)
    elif args.cv_fold==1:
        if args.datasets.lower() == 'camelyon16':
            #----------------------改----------------------------
            label_path=os.path.join('/home/liuzequn/MHIM-MIL-master/MHIM-MIL-master/camelyon16','label_c16.csv')
            p, l,train_patiences_list,train_patiences_label_list,test_patiences_list,test_patiences_label_list = get_patient_label_no_cross_fold(label_path)
            index = [i for i in range(len(train_patiences_list))]
            train_patiences_list = train_patiences_list[index]
            train_patiences_label_list = train_patiences_label_list[index]
            index = [i for i in range(len(test_patiences_list))]
            test_patiences_list = test_patiences_list[index]
            test_patiences_label_list = test_patiences_label_list[index]
            train_p, train_l, test_p, test_l,val_p,val_l = train_patiences_list,train_patiences_label_list,test_patiences_list,test_patiences_label_list,[],[]
         
    acs, pre, rec,fs,auc,te_auc,te_fs=[],[],[],[],[],[],[]
    ckc_metric = [acs, pre, rec,fs,auc,te_auc,te_fs]
    print('Dataset: ' + args.datasets)
    for k in range(args.fold_start, args.cv_fold):
        if args.datasets == 'tcga':
            if k !=3:
                continue
        ckc_metric = one_fold(args,k,ckc_metric,train_p, train_l, test_p, test_l,val_p,val_l)
        
def one_fold(args,k,ckc_metric,train_p, train_l, test_p, test_l,val_p,val_l):
    # ---> Initialization
    seed_torch(args.seed)
    loss_scaler = GradScaler() if args.amp else None
    amp_autocast = torch.cuda.amp.autocast if args.amp else suppress
    #device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    acs,pre,rec,fs,auc,te_auc,te_fs = ckc_metric
    # ---> Loading data
    if args.datasets.lower() == 'camelyon16':
        train_set = C16Dataset_no_cross_fold(train_p,train_l,root=args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize,is_train=True)
        test_set = C16Dataset_no_cross_fold(test_p,test_l,root=args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize)
        if args.val_ratio != 0.:
            val_set = C16Dataset_no_cross_fold(val_p,val_l,root=args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize)
        else:
            val_set = test_set
    elif args.datasets.lower() == 'tcga':

        train_set = TCGADataset(train_p[k],train_l[k],args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize,is_train=True)
        test_set = TCGADataset(test_p[k],test_l[k],args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize)
        if args.val_ratio != 0.:
            val_set = TCGADataset(val_p[k],val_l[k],args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize)
        else:
            val_set = test_set

    if args.fix_loader_random:
        # generated by int(torch.empty((), dtype=torch.int64).random_().item())
        big_seed_list = 7784414403328510413
        generator = torch.Generator()
        generator.manual_seed(big_seed_list)  
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,generator=generator)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=RandomSampler(train_set), num_workers=args.num_workers)

    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

   
    if args.model == 'SD-MIL':
        model_params = {
            'input_dim': args.input_dim,
            'n_classes': args.n_classes,
            'dropout': args.dropout,
            'act': args.act,
            'region_num': args.region_num,
            'pos': args.pos,
            'pos_pos': args.pos_pos,
            'pool': args.pool,
            'peg_k': args.peg_k,
            'drop_path': args.drop_path,
            'n_layers': args.n_trans_layers,
            'n_heads': args.n_heads,
            'attn': args.attn,
            'da_act': args.da_act,
            'trans_dropout': args.trans_drop_out,
            'ffn': args.ffn,
            'mlp_ratio': args.mlp_ratio,
            'trans_dim': args.trans_dim,
            'epeg': args.epeg,
            'min_region_num': args.min_region_num,
            'qkv_bias': args.qkv_bias,
            'epeg_k': args.epeg_k,
            'epeg_2d': args.epeg_2d,
            'epeg_bias': args.epeg_bias,
            'epeg_type': args.epeg_type,
            'region_attn': args.region_attn,
            'peg_1d': args.peg_1d,
            'cr_msa': args.cr_msa,
            'crmsa_k': args.crmsa_k,
            'all_shortcut': args.all_shortcut,
            'crmsa_mlp':args.crmsa_mlp,
            'crmsa_heads':args.crmsa_heads,
            'model_dowan_aggregate_model':args.model_dowan_aggregate_model,
         }
        model = rrt.RRTMIL(**model_params).to(device)
    if args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()

    # optimizer
    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_sche == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, 0) if not args.lr_supi else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch*len(train_loader), 0)
    elif args.lr_sche == 'step':
        assert not args.lr_supi
        # follow the DTFD-MIL
        # ref:https://github.com/hrzhang1123/DTFD-MIL
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,args.num_epoch / 2, 0.2)
    elif args.lr_sche == 'const':
        scheduler = None

    if args.early_stopping:
        early_stopping = EarlyStopping(patience=30 if args.datasets=='camelyon16' else 20, stop_epoch=args.max_epoch if args.datasets=='camelyon16' else 100,save_best_model_stage=np.ceil(args.save_best_model_stage * args.num_epoch))
    else:
        early_stopping = None

    if args.fix_train_random:
        seed_torch(args.seed)
    # test
    if not args.no_log:
        #Camelyon16 resnet50
        #best_std = torch.load('./weight/Camelyon16_resnet50.pt/',map_location='cpu')
        #Camelyon16 ctranspath
        best_std = torch.load('./weights/Camelyon16_ctranspath.pt',map_location='cpu')
        #TCGA resnet50
        #best_std = torch.load('./weights/TCGA_resnet50.pt',map_location='cpu')
        #TCGA ctranspath
        #best_std = torch.load('./weights/TCGA_ctranspath',map_location='cpu')
        info = model.load_state_dict(best_std['model'],strict=False)
        print(info)
    accuracy, auc_value, precision, recall, fscore = test_2(args,model,test_loader,device,criterion)
    print('accuracy, auc_value, precision, recall, fscore:',accuracy, auc_value, precision, recall, fscore)
    return [accuracy,precision,recall,fs,fscore,auc_value]

def five_scores_2(bag_labels, bag_predictions,sub_typing=False):
    fpr, tpr, threshold = roc_curve(bag_labels, bag_predictions, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    # threshold_optimal=0.5
    auc_value = roc_auc_score(bag_labels, bag_predictions)
    this_class_label = np.array(bag_predictions)
    this_class_label[this_class_label>=threshold_optimal] = 1
    this_class_label[this_class_label<threshold_optimal] = 0
    bag_predictions = this_class_label
    avg = 'macro' if sub_typing else 'binary'
    precision, recall, fscore, _ = precision_recall_fscore_support(bag_labels, bag_predictions, average=avg)
    accuracy = accuracy_score(bag_labels, bag_predictions)
    return accuracy, auc_value, precision, recall, fscore
    
    
def test_2(args,model,loader,device,criterion):
    model.eval()
    test_loss_log = 0.
    bag_logit, bag_labels=[], []
    #1111111111111111111111111111111111111111111
    all_patient_names = []
    all_bag=[]
    with torch.no_grad():
        for i, data in enumerate(loader):
            if len(data[1]) > 1:
                bag_labels.extend(data[1].tolist())
            else:
                bag_labels.append(data[1].item())  
            if isinstance(data[0],(list,tuple)):
                for i in range(len(data[0])):
                    data[0][i] = data[0][i].to(device)
                bag=data[0].to(device)
                batch_size=data[0][0].size(0)
            else:
                bag=data[0].to(device)  # b*n*1024
                batch_size=bag.size(0)
            label=data[1].to(device)
            if args.model == 'dsmil':
                test_logits,_ = model(bag)
            elif args.model_dowan_aggregate_model=='dsmil':
                test_logits,max_prediction = model(bag,label,loss=criterion)
            else:
                test_logits = model(bag)
                
            if args.loss == 'ce':
                if (args.model == 'dsmil' and args.ds_average) or (args.model == 'mhim' and isinstance(test_logits,(list,tuple))):
                    test_loss = criterion(test_logits[0].view(batch_size,-1),label)
                    bag_logit.append((0.5*torch.softmax(test_logits[1],dim=-1)+0.5*torch.softmax(test_logits[0],dim=-1))[:,1].cpu().squeeze().numpy())
                elif args.model_dowan_aggregate_model=='dsmil' and args.ds_average:
                    test_loss = criterion(test_logits[0].view(batch_size,-1),label)
                    bag_logit.append((0.5*torch.softmax(test_logits[1],dim=-1)+0.5*torch.softmax(test_logits[0],dim=-1))[:,1].cpu().squeeze().numpy())
                
                else:
                    test_loss = criterion(test_logits.view(batch_size,-1),label)
                    if batch_size > 1:
                        bag_logit.extend(torch.softmax(test_logits,dim=-1)[:,1].cpu().squeeze().numpy())
                    else:
                        bag_logit.append(torch.softmax(test_logits,dim=-1)[:,1].cpu().squeeze().numpy())
            elif args.loss == 'bce':
                if args.model == 'dsmil' and args.ds_average :
                    test_loss = criterion(test_logits[0].view(batch_size,-1),label)
                    bag_logit.append((0.5*torch.sigmoid(test_logits[1])+0.5*torch.sigmoid(test_logits[0]).cpu().squeeze().numpy()))
                elif args.model_dowan_aggregate_model=='dsmil' and args.ds_average:
                    bag_loss = criterion(test_logits.view(1,-1), label.view(1,-1).float())
                    max_loss = criterion(max_prediction.view(1,-1), label.view(1,-1).float())
                    test_loss = 0.5*bag_loss + 0.5*max_loss
                    test_loss = test_loss.mean()
                    bag_logit.extend([torch.sigmoid(test_logits).squeeze().cpu().numpy()])
                else:
                    test_loss = criterion(test_logits.view(batch_size,-1),label.view(1,-1).float())
                    bag_logit.append(torch.sigmoid(test_logits).cpu().squeeze().numpy())
            
    
   
    accuracy, auc_value, precision, recall, fscore = five_scores_2(bag_labels, bag_logit, sub_typing=not args.datasets.lower() == 'camelyon16')
    
    return accuracy, auc_value, precision, recall, fscore

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MIL Training Script')

    # Dataset 
    parser.add_argument('--datasets', default='camelyon16', type=str, help='[camelyon16, tcga]')
    parser.add_argument('--dataset_root', default='/data/xxx/TransMIL', type=str, help='Dataset root path')
    parser.add_argument('--tcga_max_patch', default=-1, type=int, help='Max Number of patch in TCGA [-1]')
    parser.add_argument('--fix_loader_random', action='store_true', help='Fix random seed of dataloader')
    parser.add_argument('--fix_train_random', action='store_true', help='Fix random seed of Training')
    parser.add_argument('--val_ratio', default=0., type=float, help='Val-set ratio')
    parser.add_argument('--fold_start', default=0, type=int, help='Start validation fold [0]')
    parser.add_argument('--cv_fold', default=3, type=int, help='Number of cross validation fold [3]')
    parser.add_argument('--persistence', action='store_true', help='Load data into memory') 
    parser.add_argument('--same_psize', default=0, type=int, help='Keep the same size of all patches [0]')
    parser.add_argument('--tcga_sub', default='nsclc', type=str, help='[nsclc,brca]')
    parser.add_argument("--local_rank", default=-1,type=int)
    parser.add_argument("--num_gpus ", default=3,type=int)
    # Train
    parser.add_argument('--cls_alpha', default=1.0, type=float, help='Main loss alpha')
    parser.add_argument('--aux_alpha', default=1.0, type=float, help='Auxiliary loss alpha')
    parser.add_argument('--auto_resume', action='store_true', help='Resume from the auto-saved checkpoint')
    parser.add_argument('--num_epoch', default=200, type=int, help='Number of total training epochs [200]')
    parser.add_argument('--early_stopping', action='store_false', help='Early stopping')
    parser.add_argument('--max_epoch', default=130, type=int, help='Number of max training epochs in the earlystopping [130]')
    parser.add_argument('--input_dim', default=1024, type=int, help='dim of input features. PLIP features should be [512]')
    parser.add_argument('--n_classes', default=2, type=int, help='Number of classes')
    parser.add_argument('--batch_size', default=1, type=int, help='Number of batch size')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers in the dataloader')
    parser.add_argument('--loss', default='ce', type=str, help='Classification Loss [ce, bce]')
    parser.add_argument('--opt', default='adam', type=str, help='Optimizer [adam, adamw]')
    parser.add_argument('--save_best_model_stage', default=0., type=float, help='See DTFD')
    parser.add_argument('--model', default='rrtmil', type=str, help='Model name')
    parser.add_argument('--seed', default=2021, type=int, help='random number [2021]' )
    parser.add_argument('--lr', default=2e-4, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--lr_sche', default='cosine', type=str, help='Deacy of learning rate [cosine, step, const]')
    parser.add_argument('--lr_supi', action='store_true', help='LR scheduler update per iter')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--accumulation_steps', default=1, type=int, help='Gradient accumulate')
    parser.add_argument('--clip_grad', default=.0, type=float, help='Gradient clip')
    parser.add_argument('--always_test', action='store_true', help='Test model in the training phase')

    # Model
    # Other models
    parser.add_argument('--ds_average', action='store_true', help='DSMIL hyperparameter')
    # Our
    parser.add_argument('--only_rrt_enc',action='store_true', help='RRT+other MIL models [dsmil,clam,]')
    parser.add_argument('--act', default='relu', type=str, help='Activation func in the projection head [gelu,relu]')
    parser.add_argument('--dropout', default=0.25, type=float, help='Dropout in the projection head')
    # Transformer
    parser.add_argument('--attn', default='rmsa', type=str, help='Inner attention')
    parser.add_argument('--pool', default='attn', type=str, help='Classification poolinp. use abmil.')
    parser.add_argument('--ffn', action='store_true', help='Feed-forward network. only for ablation')
    parser.add_argument('--n_trans_layers', default=2, type=int, help='Number of layer in the transformer')
    parser.add_argument('--mlp_ratio', default=4., type=int, help='Ratio of MLP in the FFN')
    parser.add_argument('--qkv_bias', action='store_false')
    parser.add_argument('--all_shortcut', action='store_true', help='x = x + rrt(x)')
    # R-MSA
    parser.add_argument('--region_attn', default='native', type=str, help='only for ablation')
    parser.add_argument('--min_region_num', default=0, type=int, help='only for ablation')
    parser.add_argument('--region_num', default=8, type=int, help='Number of the region. [8,12,16,...]')
    parser.add_argument('--trans_dim', default=64, type=int, help='only for ablation')
    parser.add_argument('--n_heads', default=8, type=int, help='Number of head in the R-MSA')
    parser.add_argument('--trans_drop_out', default=0.1, type=float, help='Dropout in the R-MSA')
    parser.add_argument('--drop_path', default=0., type=float, help='Droppath in the R-MSA')
    # PEG or PPEG. only for alation
    parser.add_argument('--pos', default='none', type=str, help='Position embedding, enable PEG or PPEG')
    parser.add_argument('--pos_pos', default=0, type=int, help='Position of pos embed [-1,0]')
    parser.add_argument('--peg_k', default=7, type=int, help='K of the PEG and PPEG')
    parser.add_argument('--peg_1d', action='store_true', help='1-D PEG and PPEG')
    # EPEG
    parser.add_argument('--epeg', action='store_false', help='enable epeg')
    parser.add_argument('--epeg_bias', action='store_false', help='enable conv bias')
    parser.add_argument('--epeg_2d', action='store_true', help='enable 2d conv. only for ablation')
    parser.add_argument('--epeg_k', default=15, type=int, help='K of the EPEG. [9,15,21,...]')
    parser.add_argument('--epeg_type', default='attn', type=str, help='only for ablation')
    # CR-MSA
    parser.add_argument('--cr_msa', action='store_false', help='enable CR-MSA')
    parser.add_argument('--crmsa_k', default=3, type=int, help='K of the CR-MSA. [1,3,5]')
    parser.add_argument('--crmsa_heads', default=8, type=int, help='head of CR-MSA. [1,8,...]')
    parser.add_argument('--crmsa_mlp', action='store_true', help='mlp phi of CR-MSA?')

    # DAttention
    parser.add_argument('--da_act', default='relu', type=str, help='Activation func in the DAttention [gelu,relu]')

    # Shuffle
    parser.add_argument('--patch_shuffle', action='store_true', help='2-D group shuffle')
    parser.add_argument('--group_shuffle', action='store_true', help='Group shuffle')
    parser.add_argument('--shuffle_group', default=0, type=int, help='Number of the shuffle group')

    # Misc
    parser.add_argument('--title', default='default', type=str, help='Title of exp')
    parser.add_argument('--project', default='mil_new_c16', type=str, help='Project name of exp')
    parser.add_argument('--log_iter', default=100, type=int, help='Log Frequency')
    parser.add_argument('--amp', action='store_true', help='Automatic Mixed Precision Training')
    parser.add_argument('--wandb', action='store_true', help='Weight&Bias')
    parser.add_argument('--no_log', action='store_true', help='Without log')
    parser.add_argument('--model_path', type=str, help='Output path')
    parser.add_argument('--model_dowan_aggregate_model', type=str, default='rmsa' )
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
    if not os.path.exists(os.path.join(args.model_path,args.project)):
        os.mkdir(os.path.join(args.model_path,args.project))
    args.model_path = os.path.join(args.model_path,args.project,args.title)
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    if args.datasets == 'camelyon16':
        args.fix_loader_random = True
        args.fix_train_random = True

    if args.datasets == 'tcga':
        args.num_workers = 0
        args.always_test = True


        
    print(args)
    localtime = time.asctime( time.localtime(time.time()) )
    print(localtime)
    main(args=args)


#CUDA_VISIBLE_DEVICES=3 python3 test2.py --project=SD-MIL --datasets=camelyon16 --dataset_root=./datasets/Camelyon16/ResNet50 --model_path=test --cv_fold=1 --model=SD-MIL --pool=attn --n_trans_layers=2 --da_act=tanh --title=camelyon16_resnet50 --epeg_k=15 --crmsa_k=1 --all_shortcut --seed=2021 --model_dowan_aggregate_model='abmil'
#CUDA_VISIBLE_DEVICES=3 python3 test2.py --project=SD-MIL --datasets=camelyon16 --dataset_root=./datasets/Camelyon16/CtransPath --model_path=test --cv_fold=1 --model=myself_rrtmil --pool=attn --n_trans_layers=2 --da_act=tanh --title=camelyon16_resnet50 --epeg_k=15 --crmsa_k=1 --all_shortcut --seed=2021 --model_dowan_aggregate_model='abmil' --input_dim=768


#CUDA_VISIBLE_DEVICES=2 python3 test2.py --project=SD-MIL --datasets=tcga --dataset_root=./datasets/TCGA/ResNet50 --model_path=test  --cv_fold=4 --model=myself_rrtmil --pool=attn --n_trans_layers=2 --da_act=tanh --title=tcga_resnet50 --epeg_k=15 --crmsa_k=1 --all_shortcut --seed=2021 --model_dowan_aggregate_model='abmil'  --val_ratio=0.13
#CUDA_VISIBLE_DEVICES=2 python3 test2.py --project=SD-MIL --datasets=tcga --dataset_root=./datasets/TCGA/CtransPath --model_path=test  --cv_fold=4 --model=myself_rrtmil --pool=attn --n_trans_layers=2 --da_act=tanh --title=tcga_ctranspath --epeg_k=15 --crmsa_k=1 --all_shortcut --seed=2021 --model_dowan_aggregate_model='abmil'  --val_ratio=0.13 --input_dim=768