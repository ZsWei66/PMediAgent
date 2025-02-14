from openai import OpenAI
import os
import logging
from transformers import AutoModel, AutoTokenizer
import torch

logging.basicConfig(level=logging.INFO)

# ChatLaw
model = AutoModel.from_pretrained("pandalla/ChatLaw2E_plain_7B", trust_remote_code=True)
model_name = "pandalla/ChatLaw2E_plain_7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_text(prompt, max_length=4096):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Deepseek API
client = OpenAI(api_key="", base_url="")

TP_prompt = """
## Role
你是一位拥有二十年从业经验的矛盾纠纷调解员，擅长分析修改矛盾纠纷案例。你的任务是阅读矛盾纠纷案例，修改重写原案例文本。

## Task:
1)将案例中的调解员人设统一为男性调解员，去除调解员的职务、性别、名字等信息
2)将案例中提及的调解方法部分的描述进行删除
3)将文本统一结构为（案例标题，案情经过介绍，调解过程经过，适用法律）

## Constraints
请仔细阅读原文本并将其中对应内容重写为统一结构，修改完毕后重新输出重写的案例文本即可，不需要额外输出。

## 请处理以下案例：
{}
"""

MS_prompt = """
## Role
你是一位拥有二十年从业经验的矛盾纠纷调解员，擅长分析矛盾纠纷案例。你负责基于矛盾纠纷历史案例记录，设计记录笔记。

## Task：
- 解析纠纷案例记录，找出所有人物（当事人、辅助调解员）。
- 解析纠纷案例记录，找出最终达成的调解方案。

## Constraints
- 你的输出格式如下：

所有人物：
纠纷当事人1：他的名字
...
纠纷当事人n：他的名字

辅助调解人员1：他的名字
....
辅助调解人员m：他的名字

调解方案：
调解方案的内容

- 注意n,m为纠纷当事人数量和辅助调解人员的数量。除上述列表不需要额外输出。

##请分析以下矛盾纠纷案例，设计咨询笔记。
{}
"""

Strategies_List = """
### Strategy 1: 了解基本情况
- 调解员对当事人进行基本情况的询问，以全面掌握当事人的背景信息。
- 通过了解当事人的基本情况，为后续的调解工作提供必要的信息支持。

### Strategy 2: 动员多种力量协助
- 动用与当事人联系密切的亲友以及相关社会力量，能够了解根本病症所在。
- 通过多种力量的协助，形成合力，共同推动调解工作的顺利进行。

### Strategy 3: 法治与德治相结合
- 调解工作以适用的法律为准绳，确保调解过程的合法性和公正性。
- 结合运用道德的要求，增强当事人的道德意识，促进问题的解决。

### Strategy 4: 抓住主要矛盾调解
- 抓住事件的主要矛盾，突出调解工作的重点，避免被次要问题分散注意力。
- 集中精力解决主要矛盾，能够有效推动调解进程，提高调解效率。

### Strategy 5: 解决思想问题与解决实际问题相结合
- 通过解决当事人所面对的实际问题，来解决思想问题以达到调节效果。
- 实际问题的解决有助于缓解当事人的心理压力，为思想问题的解决创造条件。
 
### Strategy 6: 换位思考
- 设身处地地站在当事人的角度提出可以接受的调解方案，增强方案的可接受性。
- 通过换位思考，理解当事人的真实需求，提高调解方案的针对性和有效性。

### Strategy 7: 苗头预测和遏制
- 抓住带有苗头性、倾向性的问题，把纠纷遏制在萌芽状态，防止矛盾扩大和深化。
- 通过及时发现和处理苗头性问题，避免问题升级，维护社会稳定。

### Strategy 8: 模糊处理法
- 对于一些非原则性问题，采取淡化、隐去，对问题采取“点到为止”。
- 模糊处理法能够达到既调解了矛盾问题，又保护当事人自尊的效果。

### Strategy 9: 褒扬激励的方法
- 对于当事人的优点长处进行表扬鼓励，从而调动当事人积极性。
- 通过褒扬激励，增强当事人的自信心，促进其积极配合调解工作。

### Strategy 10: 达成调解协议
- 明确协议内容，确保各方就争议事项达成一致，明确各自的权利和义务。

### Strategy 11: 其他
- 多用于一些无内容的口语化的策略，例如问好告别等的内容属于此类调解策略
"""

DR_prompt = """
## Role
你是一位拥有二十年从业经验的基层矛盾调解员，擅长重建基层矛盾调解对话场景。你负责基于历史案例记录，还原所有当事人和调解员的多轮长对话。

## Task
- 对于调解员轮次的话语，你能够结合上下文对话内容以及调解事件描述的真实内容
，还原调解员真实策略选择，并且只能从下面的十一种策略中进行选择。
- 对于非调解员轮次的话语，能够还原出他们的真实表达

## Constraints
- 输出格式如下：
调解员轮次的话语输出格式：调解员（调解策略）：“调解话语”，调解策略是你选择的十一个之一，调解话语是基于此策略的对话内容
当事人以及辅助调解人员轮次的话语输出格式：当事人名字：当事人话语

- 调解通常以“了解基本情况”开始，最终“达成调解协议”。请保证剧情的完整性，不要出现非对话的文本内容，不要漏掉对话细节。

## 可使用的调解策略包括：
{}

## 所涉及的人员包含：
{}

## 达成的协议如下：
{}

## 法律顾问给出的适用法律参考：
{}

## 请参考上述提供的信息还原以下矛盾纠纷案例的调解对话场景
{}
"""

LC_prompt = """
{}
什么法律适用如上案件？
"""

def get_completion(sys_prompt="", usr_prompt, temperature=0.7):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": usr_prompt},
            ],
            temperature=temperature,  
            frequency_penalty=0.02,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error during API call: {e}")
        return None

def process_case(case_file):
    logging.info(f"Processing file: {case_file}")
    try:
        with open(case_file, 'r', encoding='utf-8') as file:
            user_input = file.read()
    except Exception as e:
        logging.error(f"Error reading file {case_file}: {e}")
        return
    
    # TextPurier
    logging.info("Sending request to API for TP...")
    responseTP = get_completion(TP_prompt.format(user_input))
    if responseTP is None:
        return
    logging.info("Received response from API for TP.")
    print("TP Response:\n", responseTP)

    # Mediation Secretary
    logging.info("Sending request to API for MS...")
    responseMS = get_completion(MS_prompt.format(user_input))
    if responseMS is None:
        return
    logging.info("Received response from API for MS.")
    print("MS Response:\n", responseMS)

    try:
        parts = responseMS.split("调解方案：")
        all_people_str = parts[0].strip()
        mediation_agreement = parts[1].strip()
    except IndexError as e:
        logging.error(f"Error parsing response from step 1: {e}")
        return
    
    lines = all_people_str.strip().split('\n')
    people = ""
    for line in lines:
        if line.startswith("纠纷当事人") or line.startswith("辅助调解人员"):
            print(line)
            print("________")
            name = line.split('：')[1]
            print('name = ', name)
            people = people + name + ", "
    
    # Legal Counsel
    Legal_reference = generate_text(LC_prompt.format(user_input))

    # Dialogue Rebuilder
    logging.info("Sending request to API for DR...")
    responseDR = get_completion(DR_prompt.format(Strategies_List, people, mediation_agreement, Legal_reference, user_input))
    if responseDR is None:
        return
    logging.info("Received response from API for DR.")
    print("DR Response:\n", responseDR)
    return people, responseDR

def write2file(text_chat, text_parties, case_type, case_file):
    output_dirs = {
        "chat": "output_chat",
        "parties": "output_chat_parties",
        "case_type": "output_chat_case_type"
    }
    
    texts = {
        "chat": text_chat,
        "parties": text_parties,
        "case_type": case_type
    }
    
    for key, output_dir in output_dirs.items():
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_file = os.path.join(output_dir, case_file.replace('.txt', f'_{key}.txt'))
        
        try:
            with open(output_file, 'w', encoding='utf-8') as file:
                file.write(texts[key])
            logging.info(f"Saved output to {output_file}")
        except Exception as e:
            logging.error(f"Error writing to file {output_file}: {e}")    

for case_file in os.listdir('.'):
    if case_file.endswith('.txt') :
        with open(case_file, 'r', encoding='utf-8') as file:
            # first line belongs to the dispute type
            first_line = file.readline().strip()
        case_type = first_line.split('：')[1]
        parties, chat = process_case(case_file)
        write2file(chat, parties, case_type, case_file)