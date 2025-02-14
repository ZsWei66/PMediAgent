from env_pm import Env
from agent import PPDPP
from utils import *
from itertools import count
from tqdm import tqdm
import argparse
from transformers import BertTokenizer, RobertaTokenizer, BertConfig, RobertaConfig
from fastchat.model import add_model_args
import random
import subprocess
tok = {'bert': BertTokenizer, 'roberta': RobertaTokenizer}
cfg = {'bert': BertConfig, 'roberta': RobertaConfig}

def train(args, config, dataset, filename, tokenizer):
    env = Env(args, dataset, mode='train') # env init
    set_random_seed(args.seed)
    policy = PPDPP(args, config, tokenizer) # policy network init

    # load policy parameters
    if args.sft_dir is not None and args.prompt_type == 'ppdpp':
        # sft 
        print('Staring loading policy model from {}'.format(args.sft_dir))
        policy.load_model(data_name=args.data_name, filename=args.sft_dir)
        
    if args.load_rl_epoch > 0:
        # RL
        print('Staring loading rl model in epoch {}'.format(args.load_rl_epoch))
        policy.load_model(data_name=args.data_name, filename=filename, epoch_user=args.load_rl_epoch)

    test_performance = []
    if args.do_eval:
        SR16_mean = evaluate(args, dataset, policy, filename, 0, env) # PMediEval
        test_performance = [SR16_mean]

    if not args.do_train:
        return
    
    for train_step in range(1, args.max_steps+1):
        SR, AvgT, total_reward = 0., 0., 0.
        loss = torch.tensor(0, dtype=torch.float, device=args.device)

        for i_episode in tqdm(range(args.sample_times),desc='sampling'):
            print('\n================new tuple:{}===================='.format(i_episode))

            state = env.reset() # init
            epi_reward = 0
            done = 0

            for t in count():   
                state, reward, done = env.step(policy) 
                epi_reward += reward
                reward = torch.tensor([reward], device=args.device, dtype=torch.float)
                policy.rewards.append(reward)

                if done:
                    if done == 1:
                        SR += 1
                    AvgT += t+1
                    total_reward += epi_reward
                    break

            if len(policy.rewards) != 0:
                print('optimizing model.')
                newloss = policy.optimize_model()
                loss += newloss
            
        print('loss : {} in epoch_user {}'.format(loss.item()/args.sample_times, args.sample_times))
        print('SR:{}, AvgT:{}, rewards:{} Total epoch_user:{}'.format(SR / args.sample_times,
                    AvgT / args.sample_times, total_reward / args.sample_times, args.sample_times))
        
        if train_step % args.valid_num == 0:
            # select random 20 cases for valid
            SR_all = valid_20(args, dataset, policy, filename, train_step, env)
            test_performance.append(SR_all)
            print('Valid result for turn step', train_step)
            print('=', SR_all)
            print('\n')

        if train_step % args.save_num == 0:
            policy.save_model(data_name=args.data_name, filename=filename, epoch_user=train_step)
    
    # PMediEval
    SR_final = evaluate(args, dataset, policy, filename, 0, env)
    print('SR_final =', SR_final)   
    print(test_performance)


def valid_20(args, dataset, policy, filename, i_episode, train_env):
    test_env = Env(args, dataset, mode='valid') # env init
    set_random_seed(args.seed)
    SR, AvgT, total_reward = 0, 0, 0
    SR_turn = [0]* args.max_turn
    turn_result = []
    result = []
    test_size = len(test_env.dataset)
    
    selected_count = 20
    selected_indices = random.sample(range(test_size), selected_count)
    
    test_filename_1 = 'Evaluate-epoch-{}-'.format(i_episode)
    record_filename = 'Record-epoch-{}-'.format(i_episode) + test_filename_1
    REC_PATH = 'tmp/pm/eval_result' + record_filename + '.txt'
    if not os.path.isdir(TMP_DIR[args.data_name] + '/eval_result/'):
        os.makedirs(TMP_DIR[args.data_name] + '/eval_result/')
    rec_file = open(REC_PATH, 'w')
    
    for idx in tqdm(selected_indices, desc="评估进度"):
        print(f'\n================ 测试案例 {idx} ====================')
        epi_reward = 0
        done = 0
    
        test_env.test_num = idx  
        state = test_env.reset()  # load dataset[idx]
        
        for t in count():
            state, reward, done = test_env.step(policy)
            epi_reward += reward

            if done:
                if done == 1:  
                    SR_turn = [v+1 if i > t else v for i, v in enumerate(SR_turn)]
                    SR += 1
                total_reward += epi_reward
                AvgT += t + 1
                rec_file.write('%s\n\n' % str({'dialog': state, 'reward': epi_reward}))
                break
        
    SR_mean = SR / selected_count
    AvgT_mean = AvgT / selected_count
    reward_mean = total_reward / selected_count
    SR_all = [SR_mean, AvgT_mean, reward_mean]
    
    SRturn_all = [v / selected_count for v in SR_turn]
    
    print('success turn:{}'.format(SRturn_all))
    print('SR:{}, AvgT:{}, reward:{}'.format(SR_mean, AvgT_mean, reward_mean))
    
    return SR_all

def evaluate(args, dataset, policy, filename, i_episode, train_env):
    test_env = Env(args, dataset, mode='test') # env init
    set_random_seed(args.seed)
    
    # parallel
    if args.step == "1":
        start_idx = 0
        end_idx = 20
    elif args.step == "2":
        start_idx = 20
        end_idx = 40
    elif args.step == "3":
        start_idx = 40
        end_idx = 59
    elif args.step == "4":
        start_idx = 59
        end_idx = 78

    SR, AvgT, total_reward = 0, 0, 0
    SR_turn = [0]* args.max_turn
    turn_result = []
    result = []
    test_size = len(test_env.dataset)
    print('Test size: ', test_size)
    test_filename_1 = 'Evaluate-epoch-{}-'.format(i_episode)
    record_filename = 'Record-epoch-{}-'.format(i_episode) + test_filename_1
    REC_PATH = 'tmp/pm/eval_result' + record_filename + '.txt'
    if not os.path.isdir(TMP_DIR[args.data_name] + '/eval_result/'):
        os.makedirs(TMP_DIR[args.data_name] + '/eval_result/')
    rec_file = open(REC_PATH, 'w')

    for test_num in tqdm(range(start_idx, end_idx)):  #test_size 
        print('\n================test tuple:{}===================='.format(test_num + 1))
        epi_reward = 0
        done = 0

        test_env.test_num = test_num 
        state = test_env.reset()

        for t in count():  # user  dialog
            state, reward, done = test_env.step(policy)
            epi_reward += reward

            if done:
                if done == 1:  
                    SR_turn = [v+1 if i>t  else v for i, v in enumerate(SR_turn) ]
                    SR += 1
                total_reward += epi_reward
                AvgT += t+1

                rec_file.write('%s\n\n' % str({'dialog':state, 'reward':epi_reward}))
                break

        current_SR_mean = float(SR) / (test_num - start_idx + 1)
        current_AvgT_mean = float(AvgT) / (test_num - start_idx + 1)
        print('Current SR: {}, Current AvgT: {}'.format(current_SR_mean, current_AvgT_mean))    
    
    SR_mean = float(SR)/(end_idx - start_idx)
    AvgT_mean = float(AvgT)/(end_idx - start_idx)
    reward_mean = total_reward/(end_idx - start_idx)
    SR_all = [SR_mean, AvgT_mean, reward_mean]

    SRturn_all = [0] * args.max_turn
    for i in range(len(SRturn_all)):
        SRturn_all[i] = float(SR_turn[i])/(end_idx - start_idx)
    print('success turn:{}'.format(SRturn_all))
    print('SR:{}, AvgT:{}, reward:{}'.format(SR_mean, AvgT_mean, reward_mean))
    
    return SR_all

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-seed', type=int, default=25, help='random seed.')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus.')
    parser.add_argument('--epochs', '-me', type=int, default=50000, help='the number of RL train epoch')
    parser.add_argument('--gamma', type=float, default=0.999, help='reward discount factor.')
    parser.add_argument('--learning_rate', type=float, default=1e-6, help='learning rate.')

    parser.add_argument('--data_name', type=str, default='pm', choices=['pm'],
                        help='pm')
    parser.add_argument('--sft_dir', default='./sft/pm/bert/best_checkpoint', 
                        type=str, help="SFT model path.")
    parser.add_argument('--max_turn', type=int, default=15, help='max conversation turn')
    parser.add_argument('--load_rl_epoch', type=int, default=0, help='load agent from epoch')

    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--model_path", type=str, default="./vicuna-7b")
    parser.add_argument("--model_name", type=str, default="bert")
    parser.add_argument("--model_name_or_path", default='./roberta-zh', type=str, help="model name or path")

    parser.add_argument("--do_lower_case", action='store_false', help="Set this flag if you are using an uncased model.")

    parser.add_argument('--max_steps', type=int, default=6, help='max training steps')
    parser.add_argument('--sample_times', type=int, default=100, help='the epoch of sampling')
    parser.add_argument('--valid_num', type=int, default=1, help='the number of steps to evaluate RL model and metric')
    parser.add_argument('--save_num', type=int, default=1, help='the number of steps to save RL model and metric')

    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval.")
    
    parser.add_argument('--step', type=str, default='1', choices=['1','2','3','4'],
                        help='One of {1,2,3,4}. For parallel running.')
    parser.add_argument('--prompt_type', type=str, default='proactive', choices=['ppdpp','ppdpp_nosft','standard','proactive','proCot',"ICL_AIF"],
                        help='One of {ppdpp,ppdpp_nosft,standard,proactive,proCot,ICL_AIF}. Different prompt method.')
    add_model_args(parser)
    args = parser.parse_args()
    
    print(args.device)
    print('data_set:{}'.format(args.data_name))
    
    dataset = load_pm_env_dataset(args.data_name)
    filename = '{}-{}'.format(args.data_name,args.sft_dir)

    config = cfg[args.model_name].from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    tokenizer = tok[args.model_name].from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case, cache_dir=args.cache_dir)

    if not os.path.exists(args.sft_dir):
        print("no sft model, randomly initialize policy model")
        args.sft_dir = None

    train(args, config, dataset, filename, tokenizer)

if __name__ == '__main__':
    main()