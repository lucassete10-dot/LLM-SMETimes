import argparse
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import random
import numpy as np
import torch
import torch.distributed as dist
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.device import resolve_device

if __name__ == '__main__':
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='SMETimes')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, zero_shot_forecasting, in_context_forecasting]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='SMETimes_Llama',
                        help='model name, options: [SMETimes_Llama, SMETimes_Gpt2, SMEimes_Opt_1b]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--test_data_path', type=str, default='ETTh1.csv', help='test data file used in zero shot forecasting')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--drop_last',  action='store_true', default=False, help='drop last batch in data loader')
    parser.add_argument('--val_set_shuffle', action='store_false', default=True, help='shuffle validation set')
    parser.add_argument('--drop_short', action='store_true', default=False, help='drop too short sequences in dataset')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=672, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=576, help='label length')
    parser.add_argument('--token_len', type=int, default=96, help='token length')
    parser.add_argument('--test_seq_len', type=int, default=672, help='test seq len')
    parser.add_argument('--test_label_len', type=int, default=576, help='test label len')
    parser.add_argument('--test_pred_len', type=int, default=96, help='test pred len')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # model define
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--llm_ckp_dir', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0', help='llm checkpoints dir or Hugging Face repo id')
    parser.add_argument('--local_files_only', action='store_true', default=False, help='load LLM files from local cache/dir only')
    parser.add_argument('--mlp_hidden_dim', type=int, default=256, help='mlp hidden dim')
    parser.add_argument('--mlp_hidden_layers', type=int, default=2, help='mlp hidden layers')
    parser.add_argument('--mlp_activation', type=str, default='tanh', help='mlp activation')
    parser.add_argument('--num_experts', type=int, default=1, help='num experts')
    parser.add_argument('--lambda_reg', type=float, default=0.01, help='lambda reg')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0 if os.name == 'nt' else 10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', dest='use_amp', action='store_true', help='use automatic mixed precision training', default=True)
    parser.add_argument('--no_use_amp', '--no-use-amp', dest='use_amp', action='store_false', help='disable automatic mixed precision')
    parser.add_argument('--cosine', action='store_true', help='use cosine annealing lr', default=True)
    parser.add_argument('--tmax', type=int, default=10, help='tmax in cosine anealing lr')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--mix_embeds', action='store_true', help='mix embeds', default=True)
    parser.add_argument('--test_dir', type=str, default='./test', help='test dir')
    parser.add_argument('--test_file_name', type=str, default='checkpoint.pth', help='test file')
    
    # GPU
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--device', type=str, default='auto', help='device: auto, cpu, cuda, cuda:0, ...')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--visualize', action='store_true', help='visualize', default=True)
    args = parser.parse_args()

    if args.use_multi_gpu:
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "64209")
        hosts = int(os.environ.get("WORLD_SIZE", "8"))
        rank = int(os.environ.get("RANK", "0")) 
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        gpus = torch.cuda.device_count()
        args.local_rank = local_rank
        print(ip, port, hosts, rank, local_rank, gpus)
        backend = "nccl" if dist.is_nccl_available() and torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method=f"tcp://{ip}:{port}", world_size=hosts,
                                rank=rank)
        torch.cuda.set_device(local_rank)
    else:
        args.local_rank = 0

    device = resolve_device(args)
    args.device = str(device)
    if device.type != "cuda":
        args.use_amp = False
    elif not args.use_multi_gpu:
        torch.cuda.set_device(device.index if device.index is not None else args.gpu)
    print(f"Using device: {args.device}, AMP: {args.use_amp}")
    
    Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)
            setting = '{}_{}_{}_{}_sl{}_ll{}_tl{}_lr{}_bt{}_wd{}_hd{}_hl{}_cos{}_mix{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.seq_len,
                args.label_len,
                args.token_len,
                args.learning_rate,
                args.batch_size,
                args.weight_decay,
                args.mlp_hidden_dim,
                args.mlp_hidden_layers,
                args.cosine,
                args.mix_embeds,
                args.des, ii)
            if (args.use_multi_gpu and args.local_rank == 0) or not args.use_multi_gpu:
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            if (args.use_multi_gpu and args.local_rank == 0) or not args.use_multi_gpu:
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_sl{}_ll{}_tl{}_lr{}_bt{}_wd{}_hd{}_hl{}_cos{}_mix{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.seq_len,
            args.label_len,
            args.token_len,
            args.learning_rate,
            args.batch_size,
            args.weight_decay,
            args.mlp_hidden_dim,
            args.mlp_hidden_layers,
            args.cosine,
            args.mix_embeds,
            args.des, ii)
        exp = Exp(args) 
        exp.test(setting, test=1)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
