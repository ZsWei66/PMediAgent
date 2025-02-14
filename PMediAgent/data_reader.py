import os
import logging
import torch
import pickle
from prompt import PMAct
import json

logger = logging.getLogger(__name__)

# multi-parties, using its own name, rather than pre-defined name.
role_map = {'pm':{}} 
# PMAct People Mediation Act(Strategy), follow ppdpp's name.
act_map = {'pm' : PMAct} 

def write_pkl(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def read_pkl(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def load_and_cache_examples(args, tokenizer, evaluate=False):
    mode = args.set_name if evaluate else 'train'
    print(mode)
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'sft_{}_{}_{}_{}'.format(
        args.data_name,
        mode,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length)))

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = read_pkl(cached_features_file)
        print("Loaded number of instance:", len(features['source_ids']))
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        features = convert_to_features(args, tokenizer, mode)
        print("Loaded number of instance:", len(features['source_ids']))
    
        logger.info("Saving features into cached file %s", cached_features_file)
        write_pkl(features, cached_features_file)
    return features

def convert_to_features(args, tokenizer, mode):
    path = os.path.join(args.data_dir, '{}_{}.txt'.format(args.data_name, mode))
    act = sorted(list(act_map[args.data_name].keys()))
    print('tokenizing {}'.format(path))
    with open(path, 'r', encoding='utf-8') as infile:
        max_dia_len = 0
        avg_dia_len = []
        source_ids = []
        target_ids = []
        
        if args.data_name in ['pm']:
            for line in infile:
                sample = json.loads(line.strip('\n'))
                # event_description = sample['event_description']
                dial = sample['dialog']
                state = []
                # 512-token limit in RoBERTa, only the dialogue history is input for PPDDP
                # state.append(tokenizer.encode("案例介绍：%s" % event_description))
                for turn in dial:
                    if turn['speaker'] == 'sys':
                        speaker = "调解员"
                        dial_id = []
                        for s in state[::-1]:
                            if len(dial_id) + len(s) > args.max_seq_length:
                                break
                            dial_id = s[1:] + dial_id
                        if not state:
                            #print(f"Warning: Empty state found in dialog: {sample}")
                            source_id = [tokenizer.cls_token_id]  # Use [CLS] as the default value
                        else:
                            for s in state[::-1]:
                                if len(dial_id) + len(s) > args.max_seq_length:
                                    break
                                dial_id = s[1:] + dial_id
                            source_id = s[:1] + dial_id
                        
                        # check empty
                        if turn['strategy'] == '':
                            # print(f"Warning: Empty strategy found in dialog: {sample}")
                            target_id = 10  # Set a default value, here it is the strategy "Others"
                        else:
                            try:
                                target_id = act.index(turn['strategy'])
                            except ValueError:
                                target_id = 10
                    
                        source_ids.append(source_id[-args.max_seq_length+1:])
                        target_ids.append(target_id)
                        avg_dia_len.append(len(source_id))
                        max_dia_len = max(max_dia_len, len(source_id))
                    else:
                        speaker = turn['speaker']
                    state.append(tokenizer.encode("%s: %s" % (speaker, turn['text'])))
                    
        print('{} set, max_dia_len: {}, avg_dia_len: {}'.format(mode, max_dia_len, float(sum(avg_dia_len))/len(avg_dia_len)))
    return {'source_ids':source_ids, 'target_ids':target_ids}
