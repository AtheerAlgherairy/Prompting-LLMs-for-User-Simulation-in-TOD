# Prompting Large Language Models for User Simulation in Task-Oriented Dialogue Systems

This repo is forked from:
## Convlab-3  (Many thanks to all contributors)
A Python-based toolkit for task-oriented dialogue (TOD) systems. It provides reinforcement learning (RL) toolkit for dialog policy module and components for evaluation. For more details on ConvLab-3, see [paper](https://aclanthology.org/2023.emnlp-demo.9/).

To duplicate the code in paper: Prompting Large Language Models for User Simulation in Task-Oriented Dialogue Systems, follow the steps here:


- [Add your Keys](#add-your-keys)
- [Playing with prompts?](#playing-with-prompts)
- [RL Configuration](#rl-configuration)
- [RL training](#rl-training)
- [Test dialogue system and Lexical diversity](#test-dialogue-system-and-lexical-diversity)
- [Citing](#citing)



## Add your Keys

Two places where you need to put your keys.. 
 
If you want to use ChatGPT simulator, add your Open AI key:
add your key in line 15 this file: `convlab/policy/chatgptSimulator/chatgpt_Simulator.py` 

If you want to use Llama simulator, add your HuggingFace access token:
add your token in line 34 in this file: `convlab/policy/llamaSimulator/llama_Simulator.py`


## Playing with prompts?

For chatgpt simulator, the prompt is given in: `convlab/policy/chatgptSimulator/Prompt_generator.py`

For Llama simulator, the prompt is given in: `convlab/policy/llamaSimulator/Prompt_generator.py`

We use the same prompt for both models, you can modify it and try other prompts..


## RL Configuration

Before you start RL training using PPO, you need to prepare the config file. 
The config files contains: epochs, number of training dialogues per epoch, number of evaluation dialogues, etc.
All config files related to RL PPO are in: `convlab/policy/ppo/configs/`

we have add two config files: 
* For chatgpt simulator: `chatgptSimulator.json`
* For llama simulator: `llamaSimulator.json`

Note: For better performance, imitating learning is done before RL. In the config files, we add a path for pretrained model (MLE) to initialize the policy network for PPO: 
`convlab/policy/mle/experiments/experiment_2023-06-15-14-44-04/save/supervised`



## RL training

Go to `convlab/policy/ppo` folder and run "train_ppo.py" with the corresponding config file.

For training with chatgpt simulator:
`
%run train_ppo.py --config_name="chatgptSimulator"
`
For training with llama simulator:
`
%run train_ppo.py --config_name="llamaSimulator"
`

## Test dialogue system and Lexical diversity

For cross model evaluation, you need to specify the trained policy model and the type of simulator for testing.

Go to `convlab/` folder and run "Test_end2end.py.py" with the following arguments:

* policy_file: the path to the trained policy model (without pol.mdl), it can be found in: `policy/ppo/finished_experiments/../save/best_ppo`
* simulator_type: choose the type of simulator to be used in testing: abus, chatgpt, or llama
* folder_name: the folder for the evaluation results, you will find it under:  `\ConvLab-3\convlab\results`
* num_of_dialogs: number of dialogues used for evaluation 

`%run test_end2end.py --policy_file="../policy/ppo/finished_experiments/../save/best_ppo"  --simulator_type="chatgpt" --folder_name="Testing_using_chatgpt" --num_of_dialogs=500`

Under `\ConvLab-3\convlab\results\Testing_using_chatgpt` (based on your "folder name"), you will find the followings:

* res.txt :  for Complete Rate, Success Rate, Precision/Recall/F1 , Dialogue Turns, Successful Dialogue Turns
* lexical_diversity.txt: for text quality metrics as described in the paper.

## Citing





