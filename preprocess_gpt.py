import argparse
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import torch
from models.Preprocess_Gpt2 import Model
from data_provider.data_loader import Dataset_Preprocess
from torch.utils.data import DataLoader
from utils.device import resolve_device

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMETimes Preprocess')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--device', type=str, default='auto', help='device: auto, cpu, cuda, cuda:0, ...')
    parser.add_argument('--use_amp', dest='use_amp', action='store_true', default=True, help='use fp16 on CUDA')
    parser.add_argument('--no_use_amp', '--no-use-amp', dest='use_amp', action='store_false', help='disable fp16')
    parser.add_argument('--local_files_only', action='store_true', default=False, help='load LLM files from local cache/dir only')
    parser.add_argument('--max_token_length', type=int, default=256, help='max tokenizer length for timestamp prompts')
    parser.add_argument('--batch_size', type=int, default=128, help='preprocess batch size')
    parser.add_argument('--num_workers', type=int, default=0 if os.name == 'nt' else 4, help='data loader num workers')
    parser.add_argument('--llm_ckp_dir', type=str, default='openai-community/gpt2-large', help='llm checkpoints dir or Hugging Face repo id')
    parser.add_argument('--dataset', type=str, default='ETTh1', 
                        help='dataset to preprocess, options:[ETTh1, ETTh2, ETTm1, ETTm2, electricity, weather, traffic]')
    args = parser.parse_args()
    args.use_multi_gpu = False
    args.local_rank = 0
    device = resolve_device(args)
    args.device = str(device)
    if device.type != 'cuda':
        args.use_amp = False
    print(args.dataset)
    
    model = Model(args)

    seq_len = 672
    label_len = 576
    pred_len = 96
    
    assert args.dataset in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'electricity', 'weather', 'traffic']
    if args.dataset == 'ETTh1':
        data_set = Dataset_Preprocess(
            root_path='./dataset/ETT-small/',
            data_path='ETTh1.csv',
            size=[seq_len, label_len, pred_len])
        
    elif args.dataset == 'ETTh2':
        data_set = Dataset_Preprocess(
            root_path='./dataset/ETT-small/',
            data_path='ETTh2.csv',
            size=[seq_len, label_len, pred_len])
        
    elif args.dataset == 'ETTm1':
        data_set = Dataset_Preprocess(
            root_path='./dataset/ETT-small/',
            data_path='ETTm1.csv',
            size=[seq_len, label_len, pred_len])
        
    elif args.dataset == 'ETTm2':
        data_set = Dataset_Preprocess(
            root_path='./dataset/ETT-small/',
            data_path='ETTm2.csv',
            size=[seq_len, label_len, pred_len])
        
    elif args.dataset == 'electricity':
        data_set = Dataset_Preprocess(
            root_path='./dataset/electricity/',
            data_path='electricity.csv',
            size=[seq_len, label_len, pred_len])
    elif args.dataset == 'weather':
        data_set = Dataset_Preprocess(
            root_path='./dataset/weather/',
            data_path='weather.csv',
            size=[seq_len, label_len, pred_len])
    elif args.dataset == 'traffic':
        data_set = Dataset_Preprocess(
            root_path='./dataset/traffic/',
            data_path='traffic.csv',
            size=[seq_len, label_len, pred_len])

    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    from tqdm import tqdm
    print(len(data_set.data_stamp))
    print(data_set.tot_len)
    save_path = os.path.join(data_set.root_path, f"{data_set.data_path.split('.')[0]}.pt")
    output_list = []
    with torch.no_grad():
        for idx, data in tqdm(enumerate(data_loader)):
            output = model(data)
            output_list.append(output.detach().cpu())
    result = torch.cat(output_list, dim=0)
    print(result.shape)
    torch.save(result, save_path)
    print(f"saved {save_path}")
