from copy import deepcopy
from transformers import pipeline, AutoTokenizer, AutoModel, PretrainedConfig, LlamaTokenizer
from convlab.policy.llamaSimulator.Prompt_generator import createLanguagePromptFromGoal
from convlab.nlu.jointBERT.unified_datasets import BERTNLU
from convlab.util import relative_import_module_from_unified_datasets
from convlab.task.multiwoz.goal_generator import GoalGenerator
from convlab.policy.policy import Policy
from convlab.policy.genTUS.unify.Goal import Goal
from convlab.nlg.template.multiwoz import TemplateNLG
from convlab.util.unified_datasets_util import load_dataset
import json
import os
import torch
import time


class UserPolicyLlama(Policy):
    """ The user policy model by Llama Derived from the UserPolicy class """

    def __init__(self, dataset, max_turn=40, **kwargs):
        """
        Constructor for UserPolicyLlama class.
        """
        self.max_turn = max_turn
        self.goal_gen = GoalGenerator()

        self.time_step = 0
        self.goal = None
        self.prompt = None
        self.goal_message = None
        self.reward = None

    # ====================LLAMA ========================
        token="Add your hf token"
        self.model_id = 'meta-llama/Llama-2-7b-chat-hf'
        self. DEFAULT_GENERATION_KWARGS = {
            "do_sample": True, "max_new_tokens": 256}
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, token=token)
        self. pipeline = pipeline(
            "text-generation",
            model=self.model_id,
            token="token,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device_map={"": 0}
        )

        self.nlg_sys = TemplateNLG(is_user=False, mode='manual')
        # We need this to transform generated text into DA to be checked by evaluator..
        self.nlu_user = BERTNLU(mode="all", config_file="multiwoz21_all.json",
                                model_file="https://huggingface.co/ConvLab/bert-base-nlu/resolve/main/bertnlu_unified_multiwoz21_all_context0.zip")


        self.DEFAULT_SYSTEM_INSTRUCTION = " ".join([
        "Pretend you are a CUSTOMER who is looking for some services, and I am the customer service AGENT.",
        "Stay in character for every response you give me. Our conversation is based on a given GOAL delimited by triple backticks.",
        "You should not generate all the information in the goal at once.",
        "You should generate short, precise, and informative response (less than 50 tokens), corresponding to only one or two items in the goal.",
        "It's crucial not to conclude the conversation until you've inquired about all these services, and in the specified order.",
        "If and only if you achieve your goal, you can conclude the conversation by expressing gratitude and say bye.",
        "If you think the assistant can not help you or the conversation falls into a infinite loop, say bye to end the conversation.",
        "You should not generate information not explicitly presented in the GOAL",
        "You should not ask general questions like: Can you tell me more about it?",
        "You should not generate emoji or special symbols."
    ]) 


    
   


        self.system_instruction=self.DEFAULT_SYSTEM_INSTRUCTION
	#======================================================
        self.init_session()

    def _chat(self, message):
        # Append to message list to store chat history..
        self.messages.append({"role": "user", "content": message})
        messages = deepcopy(self.messages)
        assert len(messages) % 2 == 0
        assert all([m["role"] == "user" for m in messages[1::2]]) and all(
            [m["role"] == "assistant" for m in messages[2::2]])

        prompt, eos_token = self.create_chat_prompt_and_eos_token(messages)
        sequences = self.pipeline(
            prompt,
            prefix=None,
            eos_token_id=self.tokenizer.convert_tokens_to_ids(eos_token),
            **{**self.DEFAULT_GENERATION_KWARGS}
        )
        response = sequences[0]['generated_text'][len(prompt):].strip()
        self.messages.append({"role": "assistant", "content": response})
        return response

    def reset_turn(self):
        self.time_step = 0
        # delete chat history..message list
        self.messages = [
            {"role": "system", "content": self.system_instruction}]

    def init_session(self, goal=None):
        """ Build new Goal and Prompt for next session """
        self.reset_turn()

        if not goal:
            self._new_goal()
        else:
            self._read_goal(goal)

        # ==================== delete messages list and create prompt with goal
        self.system_instruction = createLanguagePromptFromGoal(self.goal)
        self.messages = [
            {"role": "system", "content": self.system_instruction}]

        self.time_step = 0
        self.terminated = False
        self.add_sys_from_reward = False
        self.semantic_action = []
        self.utterance = ""

        while True:
            try:
                self.response = self._chat(
                    "Hello, and welcome to our services. How can I assist you today? ")
                break
            except Exception as e:
                print(f"An exception occurred: {e}. Retrying in 60 seconds...")

            # Sleep for a while before making the next attempt
            time.sleep(60)

        self.utterance = self.response

        if self.utterance is None:
            self.semantic_action = []
            self.utterance = ""
        else:
            self.semantic_action = self.nlu_user.predict(self.utterance)

    def _read_goal(self, data_goal):
        self.goal = Goal(goal=data_goal)

    def _new_goal(self):
        self.goal = Goal(goal_generator=self.goal_gen)

    def create_chat_prompt_and_eos_token(self, messages):
        # Llama neeeds special tokens for prompt..

        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        BOS, EOS = "<s>", "</s>"

        messages = [{"role": messages[1]["role"], "content": B_SYS + messages[0]
                     ["content"] + E_SYS + messages[1]["content"]}] + messages[2:]

        messages_list = [
            f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
            for prompt, answer in zip(messages[::2], messages[1::2])
        ]
        messages_list.append(
            f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}")

        return "".join(messages_list), self.tokenizer.eos_token

    def get_goal(self):
        if self.goal.raw_goal is not None:
            return self.goal.raw_goal
        goal = {}
        for domain in self.goal.domain_goals:
            if domain not in goal:
                goal[domain] = {}
            for intent in self.goal.domain_goals[domain]:
                if intent == "inform":
                    slot_type = "info"
                elif intent == "request":
                    slot_type = "reqt"
                elif intent == "book":
                    slot_type = "book"
                else:
                    print("unknown slot type")
                if slot_type not in goal[domain]:
                    goal[domain][slot_type] = {}
                for slot, value in self.goal.domain_goals[domain][intent].items():
                    goal[domain][slot_type][slot] = value
        return goal

    def get_reward(self):
        return self.reward

    def predict(self, sys_act):

        # update constraint
        self.time_step += 2

        sys_nlg = self.nlg_sys.generate(sys_act)
        if not sys_nlg or sys_nlg == "":
            sys_nlg = str(sys_act)

        sys_nlg = self.nlg_sys.generate(sys_act)
        if len(sys_act) == 0 or not sys_nlg or sys_nlg == "":
            sys_nlg = "Hello, and welcome to our services. How can I assist you today? "
        else:
            sys_nlg = sys_nlg

        while True:
            try:
                self.response = self._chat(sys_nlg)
                break
            except Exception as e:
                print(f"An exception occurred: {e}. Retrying in 60 seconds..")

            # Sleep for a while before making the next attempt
            time.sleep(60)

        self.utterance = self.response
        if self.utterance is None:
            self.semantic_action = []
            self.utterance = ""
        else:
            # add context..
            context = [x["content"] for x in self.messages[2:-1]]
            self.semantic_action = self.nlu_user.predict(
                self.utterance, context=context)

        # Specify the reward
        if self._usr_terminate():
            self.terminated = True

        if self.time_step > self.max_turn:
            self.terminated = True

        if self.terminated:
            return "Thank you. Bye"

        return self.utterance

    def _usr_terminate(self):
        for act in self.semantic_action:
            if act[0] in ['thank', 'bye']:
                return True
        return False

    def is_terminated(self):
        return self.terminated

    def get_semantic_action(self):
        return self.semantic_action


class UserPolicy(Policy):
    def __init__(self, dataset="multiwoz21", **kwargs):
        self.policy = UserPolicyLlama(dataset=dataset)

    def predict(self, sys_act):
        response = self.policy.predict(sys_act)
        self.semantic_action = self.policy.semantic_action
        return response

    def init_session(self, goal=None):
        self.policy.init_session(goal)
        self.semantic_action = []

    def is_terminated(self):
        return self.policy.is_terminated()

    def get_reward(self, sys_response=None):
        return self.policy.get_reward()

    def get_semantic_action(self):
        return self.policy.get_semantic_action()

    def get_goal(self):
        if hasattr(self.policy, 'get_goal'):
            return self.policy.get_goal()
        return None


if __name__ == "__main__":

    from convlab.dialog_agent import PipelineAgent
