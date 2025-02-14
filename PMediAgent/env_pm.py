from transformers import AutoTokenizer, AutoModelForCausalLM
import subprocess
from fastchat.model import load_model, get_conversation_template
import requests
import openai
from openai import OpenAI  
from utils import *
from prompt import *
from prompt import PMAct
import nltk
import re
import time

client_mediator_sft_qwen7b = OpenAI(api_key="", base_url="")
client_mediator_nosft_qwen7b = OpenAI(api_key="", base_url="")

client_deepseek = OpenAI(
    api_key="", 
    base_url=""
)

client_qwen_max = OpenAI(
    api_key="", 
    base_url="",
)

client_gpt3_5 = OpenAI(
    api_key="", 
    base_url="",
)

class Agent:
    # parties / support personnel
    def __init__(self, name, prompt):
        self.name = name
        self.prompt = prompt
    
    def printName(self):
        print("This is {} Agent.\n".format(self.name))
        print("My prompt is", self.prompt[:50])
        
    def speak(self, context):
        while True:
            response = get_completion(completion_type="conversation", sys_prompt=parties_sys_prompt.format(self.prompt), usr_prompt=parties_usr_prompt.format(self.name, context))    
            if response is not None:
                break
            print("Retrying 'parties speak' generation...")
            # time.sleep(1) 
        return self.name + ("：") + response

class MediatorAgent:
    def __init__(self, args):
        self.name = "调解员"
        self.args = args

    def printName(self):
        print("I am ", self.name)
        
    def speak(self, context, action):
        
        if self.args.prompt_type == "ppdpp" or self.args.prompt_type == "ppdpp_nosft":
            # policy planner plugin
            print('action = ', action)
            response = get_completion(completion_type="mediator_sft", sys_prompt=mediator_sys_template, usr_prompt=mediator_usr_strat_template.format(context, CDMAct[action]), temperature=0.7)
            response = response.replace("\n", "")
            return ("调解员：" + response)
        elif self.args.prompt_type == "standard":
            response = get_completion(completion_type="mediator_no_sft", sys_prompt=mediator_sys_template_standard, usr_prompt=mediator_usr_template_standard.format(context), temperature=0.7)
            response = response.replace("\n", "")
            return ("调解员：" + response)
        elif self.args.prompt_type == "proactive":
            response_strat = get_completion(completion_type="mediator_no_sft", sys_prompt=mediator_sys_template_proactive, usr_prompt=mediator_usr_template_proactive.format(strategies_list, context), temperature=0.7)
            response_cleaned = response_strat.replace(" ", "").replace("\n", "")

            strategy_name = response_cleaned.split("：")[1]
            try:
                strat = CDMAct[strategy_name]
            except KeyError:
                print(f"策略 '{strategy_name}' 不在 CDMAct 中。")
                strat = CDMAct["其他"]  

            response_proactive = get_completion(completion_type="mediator_no_sft", sys_prompt=mediator_sys_template, usr_prompt=mediator_usr_strat_template.format(context, strat), temperature=0.7)
            response = response_proactive.replace("\n", "")

            return ("调解员：" + response)
        elif self.args.prompt_type == "proCot":
            response_strat = get_completion(completion_type="mediator_no_sft", sys_prompt=mediator_sys_template_proCot, usr_prompt=mediator_usr_template_proCot.format(strategies_list, context), temperature=0.7) 
            response_cleaned = response_strat.replace(" ", "").replace("\n", "")

            try:
                strategy_name = response_cleaned.split("调解策略是：")[1]
            except IndexError:
                strategy_name = "其他"  
                print("未找到 调解策略")

            strategy_name = re.sub(r'[^\w\s]', '', strategy_name)

            try:
                strat = CDMAct[strategy_name]
            except KeyError:
                print(f"策略 '{strategy_name}' 不在 CDMAct 中。")
                strat = CDMAct["其他"]  

            response_proactive = get_completion(completion_type="mediator_no_sft", sys_prompt=mediator_sys_template, usr_prompt=mediator_usr_strat_template.format(context, strat), temperature=0.7)
            response = response_proactive.replace("\n", "")

            return ("调解员：" + response)
        elif self.args.prompt_type == "ICL_AIF":
            response = get_completion(completion_type="mediator_no_sft", sys_prompt=mediator_sys_template_ICL_AIF, usr_prompt=mediator_usr_template_ICL_AIF.format(context), temperature=0.7)
            
            response_strat = get_completion(completion_type="mediator_no_sft", sys_prompt=mediator_sys_template_ICL_AIF_1, usr_prompt=mediator_usr_template_ICL_AIF_1.format(strategies_list, response, context), temperature=0.7)
            
            response_cleaned = response_strat.replace(" ", "").replace("\n", "")
            match = re.search(r"调解策略：(.)", response_cleaned)
            if match:
                first_char = match.group(1)  # extract first word, increase robustness
                # pair strategy
                for strategy_name_k in CDMAct.keys():
                    if strategy_name_k.startswith(first_char):
                        strategy_name = strategy_name_k
                        break
                else:
                    print("未找到匹配的策略名称")
                    strategy_name = "其他" # default strategy "others"

            try:
                strat = CDMAct[strategy_name]
            except KeyError:
                print(f"策略 '{strategy_name}' 不在 CDMAct 中。")
                strat = CDMAct["其他"]  

            response_proactive = get_completion(completion_type="mediator_no_sft", sys_prompt=mediator_sys_template, usr_prompt=mediator_usr_strat_template.format(context, strat), temperature=0.7)      
            response = response_proactive.replace("\n", "")

            return ("调解员：" + response)
        
def get_completion(completion_type, sys_prompt=None, usr_prompt=None, temperature=0.7, frequency_penalty=0):
    try:
        clients = {
            "mediator_sft": ("qwen", client_mediator_sft_qwen7b, temperature, frequency_penalty),
            "mediator_no_sft": ("gpt-3.5-turbo", client_gpt3_5, temperature, 0),
            "conversation": ("qwen-max", client, temperature, 0),
            "manager": ("qwen-max", client, 0.0, 0.1),
            "reward": ("qwen-max", client, 0.0, 0.0),
        }

        if completion_type not in clients:
            raise ValueError("Invalid completion_type. Must be one of 'mediator_sft', 'mediator_no_sft', 'conversation', 'manager', 'reward'.")

        model, client_instance, temp, freq_penalty = clients[completion_type]

        response = client_instance.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": str(sys_prompt)},
                {"role": "user", "content": str(usr_prompt)},
            ],
            temperature=temp,
            frequency_penalty=freq_penalty,
            stream=False
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"Error during API call: {e}")
        return None
        
class Env(object):
    def __init__(self, args, dataset, mode, env_model=None, env_tokenizer=None):
        
        self.args = args
        self.dataset = dataset[mode]
        self.max_turn = args.max_turn
        self.conversation = []
        self.cur_conver_step = 0
        self.parties_introduction = ""
        self.mode = mode
        self.agents = []

        self.reward_dict = {
            'cdm': {
                'worsened': -1.0,
                'same': -0.5,
                'improved': 0.5,
                'resolved': 1.0,
            },
        }
        set_random_seed(args.seed)
   
    def reset(self):
        #
        self.cur_conver_step = 0
        if self.mode == 'train':
            self.case = np.random.choice(self.dataset) 
        elif self.mode == 'test':
            self.case = self.dataset[self.test_num]
            self.test_num += 1
        elif self.mode == 'valid':
            self.case = self.dataset[self.test_num]
            
        print('event_type =', self.case['event_type'])

        for parties, prompt in self.case['prompts'].items():
            self.agents.append(Agent(parties, prompt))
            self.parties_introduction = self.parties_introduction + f"{parties}:{prompt}" + '\n'

        self.parties_introduction = self.parties_introduction + "调解员：我是负责此次纠纷的调解员"
        self.agents.append(MediatorAgent(self.args))
        
        if self.args.data_name == 'cdm':
            self.conversation = "案例简介：" + self.case['event_description'] + "\n"
        print(self.conversation)
        return self.conversation

    import subprocess

    def manage(self, context):
    # next speaker
        manager_prompt_tmp_next_speaker = manager_oringin_prompt_next_speaker.format(
            self.case['disputing_parties'], self.parties_introduction, context
        )
        while True:
            next_speaker = get_completion(completion_type="manager", usr_prompt=manager_prompt_tmp_next_speaker)
            if next_speaker is not None:
                break
            print("Retrying 'next_speaker' generation...")
            time.sleep(1)  

        return next_speaker.strip()

    def step(self, policy):
        done = 0
        reward = 0

        print('---------------step:{}-------------'.format(self.cur_conver_step))
        # print(action)
        if self.cur_conver_step == self.max_turn:
            done = -1
            print('--> Maximum number of turns reached !')
            print('turn:', self.cur_conver_step)
            print(self.conversation)
            return self.conversation, reward, done
        
        if self.cur_conver_step != 0:
            reward_outputs = get_completion(completion_type="reward", sys_prompt=reward_sys_prompt, usr_prompt=reward_usr_prompt.format(self.conversation))
            reward = self.compute_reward(reward_outputs)     

            print('reward output =', reward_outputs)
            print('reward =', reward)

            if reward == 1:
                print('--> Goal completed !')
                print('turn:', self.cur_conver_step)
                print(self.conversation)
                done = 1;
                return self.conversation, reward, done
        
        print('--> On-going !')
        
        next_speaker = self.manage(self.conversation)
        print(next_speaker)
        
        for agent in self.agents:
            if agent.name == next_speaker:
                if agent.name == "调解员":
                    if self.args.prompt_type == "ppdpp" or self.args.prompt_type == "ppdpp_nosft":
                        action = policy.select_action(self.conversation)
                        response = agent.speak(self.conversation, action)
                    else:
                        response = agent.speak(self.conversation, "")
                else:
                    response = agent.speak(self.conversation)
                print("response: ", response)
                if response:
                    self.conversation  = self.conversation + "\n" + response
                break  

        self.cur_conver_step += 1
        return self.conversation, reward, done

    def compute_reward(self, outputs):
        rewards = []
        try:
            output_list = outputs.split()
        except AttributeError:
            print("Error: 'outputs' is not a string, it's a", type(outputs))
            output_list = []  
        
        for output in output_list:
            for key in self.reward_dict[self.args.data_name]:
                if key in output.lower():
                    rewards.append(self.reward_dict[self.args.data_name][key])
                    break
        if len(rewards) == 0:
            reward = 0
        else:
            reward = sum(rewards) / len(rewards)
        print(reward) 
        return reward