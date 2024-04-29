from convlab.policy.chatgptSimulator.Prompt_generator import createLanguagePromptFromGoal
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

import openai
openai.api_key = "Add your Key here.."

# =======================================================


class UserPolicyChatGPT(Policy):
    """ The user policy model by chatGPT Derived from the UserPolicy class """

    def __init__(self, dataset, max_turn=40, **kwargs):
        """
        Constructor for UserPolicyChatGPT class.
        """
        self.max_turn = max_turn
        self.dataset = load_dataset(dataset)
        self.goal_gen = GoalGenerator()
        self.time_step = 0
        self.goal = None
        self.prompt = None
        self.reward = None

        self.nlg_sys = TemplateNLG(is_user=False, mode='manual')

        self.nlu_user = BERTNLU(mode="all", config_file="multiwoz21_all.json",
                                model_file="https://huggingface.co/ConvLab/bert-base-nlu/resolve/main/bertnlu_unified_multiwoz21_all_context0.zip")

        self.init_session(**kwargs)

    def reset_turn(self):
        self.time_step = 0

    def init_session(self, goal=None):
        """ Build new Goal and Prompt for next session """
        self.reset_turn()

        if not goal:
            self._new_goal()
        else:
            self._read_goal(goal)

        self.prompt = createLanguagePromptFromGoal(self.goal)

        self.time_step = 0
        self.terminated = False
        self.add_sys_from_reward = False
        self.semantic_action = []
        self.utterance = ""

        self.messages = []
        system_msg = self.prompt
        self.messages.append({"role": "system", "content": system_msg})

    def _read_goal(self, data_goal):
        self.goal = Goal(goal=data_goal)

    def _new_goal(self):
        self.goal = Goal(goal_generator=self.goal_gen)

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
        if len(sys_act) == 0:
            sys_nlg = "Hello, and welcome to our services. How can I assist you today?"

        self.messages.append({"role": "user", "content": sys_nlg})

        while True:
            try:
                self.response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", messages=self.messages)
                break
            except openai.error.APIError as e:
                print(f"API error occurred. {e}")

            except openai.error.APIConnectionError as e:
                print(f"Failed to connect to OpenAI API: {e}")

            except openai.error.RateLimitError as e:
                print(f"Rate limit exceeded. {e}")

            except openai.error.Timeout as e:
                print(f"Request timed out: {e}")

            except openai.error.InvalidRequestError as e:
                print(f"OpenAI API request was invalid: {e}")

            except openai.error.AuthenticationError as e:
                print(f"OpenAI API request was not authorized: {e}")

            except openai.error.PermissionError as e:
                print(f"OpenAI API request was not permitted: {e}")
            except openai.error.ServiceUnavailableError as e:
                print(f"OpenAI API request was not permitted: {e}")

            # Sleep for a while before making the next attempt
            time.sleep(30)

        reply = self.response["choices"][0]["message"]["content"]
        self.messages.append({"role": "assistant", "content": reply})
        self.utterance = reply

        context = [x["content"] for x in self.messages[2:-1]]
        self.semantic_action = self.nlu_user.predict(reply, context=context)

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
    def __init__(self,
                 dataset="multiwoz21",
                 **kwargs):
        self.policy = UserPolicyChatGPT(dataset=dataset)

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
    import os

    from convlab.dialog_agent import PipelineAgent
