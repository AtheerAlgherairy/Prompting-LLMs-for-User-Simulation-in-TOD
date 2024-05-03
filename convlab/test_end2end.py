import sys
import argparse
import random 
import numpy as np
import torch
from convlab.policy.ppo.ppo import PPO
from convlab.nlg.template.multiwoz import TemplateNLG
from convlab.nlu.jointBERT.unified_datasets.nlu import BERTNLU   
from convlab.dst.rule.multiwoz import RuleDST
from convlab.policy.rule.multiwoz import RulePolicy
from convlab.policy.vector.vector_binary import VectorBinary  
from convlab.dialog_agent import PipelineAgent, BiSession
from convlab.util.analysis_tool.analyzer import Analyzer
from pprint import pprint
import random
import numpy as np
import torch
import re
import os
from collections import Counter
import math
from statistics import mean
import random
import json
from lexical_diversity import lex_div as ld

def set_seed(r_seed):
    random.seed(r_seed)
    np.random.seed(r_seed)
    torch.manual_seed(r_seed)

def test_end2end(policy_file, type_of_simulator, folder_name, num_of_dialogs):
    seed = 20200202
    set_seed(seed)

    sys_nlu = BERTNLU(mode='all', config_file='multiwoz21_all.json', model_file="https://huggingface.co/ConvLab/bert-base-nlu/resolve/main/bertnlu_unified_multiwoz21_all_context.zip")
    sys_dst = RuleDST()
    bin_vector = VectorBinary(dataset_name='multiwoz21', character='sys', use_masking=True, manually_add_entity_names=False, seed=0)
    sys_policy = PPO(is_train=False, seed=seed, vectorizer=bin_vector)
    sys_policy.load_policy(filename=policy_file)
    sys_nlg = TemplateNLG(is_user=False)
    sys_agent = PipelineAgent(sys_nlu, sys_dst, sys_policy, sys_nlg, name='sys')

    user_nlu = BERTNLU(mode='all', config_file='multiwoz21_all.json', model_file="https://huggingface.co/ConvLab/bert-base-nlu/resolve/main/bertnlu_unified_multiwoz21_all_context.zip")
    user_dst = None
    my_user = None
    user_nlg = None

    if type_of_simulator == "llama":
        from convlab.policy.llamaSimulator.llama_Simulator import UserPolicyLlama, UserPolicy
        my_user = UserPolicyLlama(dataset="multiwoz21")
    elif type_of_simulator == "chatgpt":
        from convlab.policy.chatgptSimulator.chatgpt_Simulator import UserPolicyChatGPT, UserPolicy
        my_user = UserPolicyChatGPT(dataset="multiwoz21")
    elif type_of_simulator == "abus":
        from convlab.policy.rule.multiwoz import RulePolicy
        my_user = RulePolicy(character='usr')
        user_nlg = TemplateNLG(is_user=True)

    # Handle other simulators here

    if my_user is None:
        print("Invalid simulator type")
        return

    simulator = PipelineAgent(user_nlu, user_dst, my_user, user_nlg, name='user')
    analyzer = Analyzer(user_agent=simulator, dataset='multiwoz')
    analyzer.comprehensive_analyze(sys_agent=sys_agent, model_name=folder_name, total_dialog=num_of_dialogs)


def lexical_diversity(model_name):

        output_dir = os.path.join('results', model_name)
       
        all_log_path=os.path.join(output_dir, 'log.txt') 
        log_path_200='results/'+model_name+'/log_200.txt'

        extract_and_save_first_200_dialogues(all_log_path, log_path_200)

        log_path_200='results/'+model_name+'/log_200.txt'
        user_turns_list=extract_user_turns_from_file(log_path_200)
 
        text_quality_path=os.path.join(output_dir, 'lexical_diversity.txt')        

        with open(text_quality_path, 'w' ,  encoding="utf-8") as output_file:
                dic=get_diversity_metrics(user_turns_list)
                dic=str(dic)
                output_file.write(dic)

        print(f"Diversity metrics (perfomed on user utterances extracted from 200 dialogues ) are saved into: {text_quality_path}")

        



def extract_and_save_first_200_dialogues(input_file_path, output_file_path, num_dialogues_to_extract=200):
    with open(input_file_path, 'r' , encoding="utf-8") as file:
        content = file.read()

    dialogues = re.split(r"={2,}\n", content)
    dialogues = [dialogue.strip() for dialogue in dialogues if dialogue.strip()]

    # Take the first 'num_dialogues_to_extract' dialogues
    selected_dialogues = dialogues[:num_dialogues_to_extract]

    with open(output_file_path, 'w', encoding="utf-8") as output_file:
        for dialogue in selected_dialogues:
            output_file.write(dialogue + '\n' + '=' * 64 + '\n')

def extract_user_turns_from_file(file_path):
    with open(file_path, 'r' ,  encoding="utf-8") as file:
        content = file.read()

    dialogues = re.split(r"Dialogue ID: \d+", content)
    user_turns_list = []

    for dialogue in dialogues:
        if 'User:' in dialogue:
            user_turns = re.findall(r"User: (.+?)(?=(?:System:|$))", dialogue, re.DOTALL)
            user_turns_list.extend([user.strip() for user in user_turns])

    return user_turns_list

def get_diversity_metrics(user_responses):
    """
    in: user_responses, list
    out: richness metrics, dict
    """
    avg_lengths, total_utterances = 0, 0
    unique_grams = [Counter() for _ in range(3)]
    all_tokens = []

    for utterance in user_responses:
        # can also use ld.flemmatize which supersets ld.tokenize functionality
        tokens = ld.tokenize(utterance)
        all_tokens.extend(tokens)

        avg_lengths += len(tokens)
        total_utterances += 1

        unique_grams[0].update(tokens)
        unique_grams[1].update(
            [(a, b) for a, b in zip(tokens, tokens[1:])])
        unique_grams[2].update(
            [(a, b, c) for a, b, c in zip(
                tokens, tokens[1:], tokens[2:])])

    # 1. Number of unique uni/big/tri-grams
    # unigram count -- number of unique tokens/words among
    # all utterances
    unique_grams_count = [len(c) for c in unique_grams]

    # 2. Average utterance length
    try:
        avg_utterance_length = avg_lengths / total_utterances
    except Exception:
        avg_utterance_length = 0

    # 3. Entropy, conditional entropy
    total = sum(v for v in unique_grams[0].values())
    probs = [(u/total) for u in unique_grams[0].values()]
    entropy = -sum(p * math.log(p, 2) for p in probs)

    cond = [unique_grams[1][
        (h, w)]/unique_grams[0][h] for h, w in unique_grams[1]]
    join = [unique_grams[1][
        (h, w)]/total for h, w in unique_grams[1]]
    cond_entropy_bigram = -sum(
        j * math.log(c, 2) for c, j in zip(cond, join))

    # 4. Lexical diversity metrics from `lexical_diversity` library
    # ttr: ratio of unique to all tokens
    # ttr = ld.ttr(all_tokens)
    # also see: root_ttr, log_ttr, maas_ttr

    # mean segmental TTR, also see mattr
    msttr = ld.msttr(all_tokens, window_length=50)

    # Hypergeometric distribution D
    # A more straightforward and reliable implementation of vocD
    # (Malvern, Richards, Chipere, & Duran, 2004)
    # as per McCarthy and Jarvis (2007, 2010).
    hdd = ld.hdd(all_tokens)

    # Measure of lexical textual diversity (MTLD)
    # based on McCarthy and Jarvis (2010).
    mtld = ld.mtld(all_tokens)
    # mtld_ma_bid = ld.mtld_ma_bid(all_tokens, min=10)
    # mtld_ma_wrap = ld.mtld_ma_wrap(all_tokens, min=10)

    return {
        'total_utterances': total_utterances,
        'total_tokens': len(all_tokens),
        'num_unigrams': unique_grams_count[0],
        'num_bigrams': unique_grams_count[1],
        'num_trigrams': unique_grams_count[2],
        'avg_utterance_length': avg_utterance_length,
        'entropy': entropy,
        'cond_entropy_bigram': cond_entropy_bigram,
        'msttr': msttr,
        'hdd': hdd,
        'mtld': mtld,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test end-to-end system with different simulators.')
    parser.add_argument('--policy_file', type=str, help='Path to the policy file')
    parser.add_argument('--simulator_type', type=str, help='Type of the simulator (llama, chatgpt, abus, gentus)')
    parser.add_argument('--folder_name', type=str, help='Folder name for the model')
    parser.add_argument('--num_of_dialogs', type=int, help='Number of dialogs to analyze')
    
    args = parser.parse_args()
    
    if not all([args.policy_file, args.simulator_type, args.folder_name, args.num_of_dialogs]):
        parser.error("Please provide all arguments: --policy_file, --simulator_type, --folder_name, --num_of_dialogs")

    test_end2end(args.policy_file, args.simulator_type, args.folder_name, args.num_of_dialogs)
    lexical_diversity(args.folder_name)
