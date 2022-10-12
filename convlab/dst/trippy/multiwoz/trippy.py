# Copyright 2021 Heinrich Heine University Duesseldorf
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import json
import copy

import torch
from transformers import (BertConfig, BertTokenizer,
                          RobertaConfig, RobertaTokenizer)

from convlab.dst.trippy.multiwoz.modeling_bert_dst import (BertForDST)
from convlab.dst.trippy.multiwoz.modeling_roberta_dst import (RobertaForDST)

from convlab.dst.dst import DST
from convlab.util.multiwoz.state import default_state
from convlab.util.multiwoz.multiwoz_slot_trans import REF_SYS_DA, REF_USR_DA
from convlab.nlu.jointBERT.multiwoz import BERTNLU
from convlab.nlg.template.multiwoz import TemplateNLG
from convlab.dst.rule.multiwoz.dst_util import normalize_value

from convlab.util import relative_import_module_from_unified_datasets
ONTOLOGY = relative_import_module_from_unified_datasets('multiwoz21', 'preprocess.py', 'ontology')
TEMPLATE_STATE = ONTOLOGY['state']

import pdb
import time


MODEL_CLASSES = {
    'bert': (BertConfig, BertForDST, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForDST, RobertaTokenizer),
}


SLOT_MAP_TRIPPY_TO_UDF = {
    'hotel': {
        'pricerange': 'price range',
        'book_stay': 'book stay',
        'book_day': 'book day',
        'book_people': 'book people',
        'addr': 'address',
        'post': 'postcode',
        'price': 'price range',
        'people': 'book people'
    },
    'restaurant': {
        'pricerange': 'price range',
        'book_time': 'book time',
        'book_day': 'book day',
        'book_people': 'book people',
        'addr': 'address',
        'post': 'postcode',
        'price': 'price range',
        'people': 'book people'
    },
    'taxi': {
        'arriveBy': 'arrive by',
        'leaveAt': 'leave at',
        'arrive': 'arrive by',
        'leave': 'leave at',
        'car': 'type',
        'car type': 'type',
        'depart': 'departure',
        'dest': 'destination'
    },
    'train': {
        'arriveBy': 'arrive by',
        'leaveAt': 'leave at',
        'book_people': 'book people',
        'arrive': 'arrive by',
        'leave': 'leave at',
        'depart': 'departure',
        'dest': 'destination',
        'id': 'train id',
        'people': 'book people',
        'time': 'duration',
        'ticket': 'price',
        'trainid': 'train id'
    },
    'attraction': {
        'post': 'postcode',
        'addr': 'address',
        'fee': 'entrance fee',
        'price': 'entrance fee'
    },
    'general': {},
    'hospital': {
        'post': 'postcode',
        'addr': 'address'
    },
    'police': {
        'post': 'postcode',
        'addr': 'address'
    }
}


class TRIPPY(DST):
    def print_header(self):
        print(" _________  ________  ___  ________  ________  ___    ___ ")
        print("|\___   ___\\\   __  \|\  \|\   __  \|\   __  \|\  \  /  /|")
        print("\|___ \  \_\ \  \|\  \ \  \ \  \|\  \ \  \|\  \ \  \/  / /")
        print("     \ \  \ \ \   _  _\ \  \ \   ____\ \   ____\ \    / / ")
        print("      \ \  \ \ \  \\\  \\\ \  \ \  \___|\ \  \___|\/  /  /  ")
        print("       \ \__\ \ \__\\\ _\\\ \__\ \__\    \ \__\ __/  / /    ")
        print("        \|__|  \|__|\|__|\|__|\|__|     \|__||\___/ /     ")
        print("          (c) 2022 Heinrich Heine University \|___|/      ")
        print()

    def print_dialog(self, hst):
        #print("Dialogue %s, turn %s:" % (self.global_diag_cnt, int(len(hst) / 2) - 1))
        print("Dialogue %s, turn %s:" % (self.global_diag_cnt, self.global_turn_cnt))
        for utt in hst[:-2]:
            print("  \033[92m%s\033[0m" % (utt))
        if len(hst) > 1:
            print(" ", hst[-2])
            print(" ", hst[-1])

    def print_inform_memory(self, inform_mem):
        print("Inform memory:")
        is_all_none = True
        for s in inform_mem:
            if inform_mem[s] != 'none':
                print("  %s = %s" % (s, inform_mem[s]))
                is_all_none = False
        if is_all_none:
            print("  -")

    def eval_user_acts(self, user_act, user_acts):
        print("User acts:")
        for ua in user_acts:
            if ua not in user_act:
                print("  \033[33m%s\033[0m" % (ua))
            else:
                print("  \033[92m%s\033[0m" % (ua))
        for ua in user_act:
            if ua not in user_acts:
                print("  \033[91m%s\033[0m" % (ua))

    def eval_dialog_state(self, state_updates, new_belief_state):
        print("Dialogue state:")
        for d in self.gt_belief_state:
            print("  %s:" % (d))
            for s in new_belief_state[d]:
                is_printed = False
                is_updated = False
                if state_updates[d][s] > 0:
                    is_updated = True
                if is_updated:
                    print("\033[3m", end='')
                if new_belief_state[d][s] != self.gt_belief_state[d][s]:
                    self.global_eval_stats[d][s]['FP'] += 1
                    if self.gt_belief_state[d][s] == '':
                        print("    \033[33m%s: %s\033[0m" % (s, new_belief_state[d][s]), end='')
                    else:
                        print("    \033[91m%s: %s\033[0m (label: %s)" % (s, new_belief_state[d][s] if new_belief_state[d][s] != '' else 'none', self.gt_belief_state[d][s]), end='')
                        self.global_eval_stats[d][s]['FN'] += 1
                    is_printed = True
                elif new_belief_state[d][s] != '':
                    print("    \033[92m%s: %s\033[0m" % (s, new_belief_state[d][s]), end='')
                    self.global_eval_stats[d][s]['TP'] += 1
                    is_printed = True
                if is_updated:
                    print(" (%s)" % (self.config.dst_class_types[state_updates[d][s]]))
                elif is_printed:
                    print()

    def eval_print_stats(self):
        print("Statistics:")
        for d in self.global_eval_stats:
            for s in self.global_eval_stats[d]:
                TP = self.global_eval_stats[d][s]['TP']
                FP = self.global_eval_stats[d][s]['FP']
                FN = self.global_eval_stats[d][s]['FN']
                prec = TP / ( TP + FP + 1e-8)
                rec = TP / ( TP + FN + 1e-8)
                f1 = 2 * ((prec * rec) / (prec + rec + 1e-8))
                print("  %s %s Recall: %.2f, Precision: %.2f, F1: %.2f" % (d, s, rec, prec, f1))

    def __init__(self, model_type="roberta",
                 model_name="roberta-base",
                 model_path="",
                 nlu_path="",
                 no_eval=False,
                 no_history=False,
                 no_normalize_value=False,
                 gt_user_acts=False,
                 gt_ds=False,
                 gt_request_acts=False):
        super(TRIPPY, self).__init__()

        self.print_header()

        self.model_type = model_type.lower()
        self.model_name = model_name.lower()
        self.model_path = model_path
        self.nlu_path = nlu_path
        self.no_eval = no_eval
        self.no_history = no_history
        self.no_normalize_value = no_normalize_value
        self.gt_user_acts = gt_user_acts
        self.gt_ds = gt_ds
        self.gt_request_acts = gt_request_acts

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.config_class, self.model_class, self.tokenizer_class = MODEL_CLASSES[self.model_type]
        self.config = self.config_class.from_pretrained(self.model_path, local_files_only=True) # TODO: parameterize
        # TODO: update config (parameters)

        # For debugging only
        path = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        path = os.path.join(path, 'data/multiwoz/value_dict.json')
        self.value_dict = json.load(open(path))

        self.ds_aux = {slot: torch.tensor([0]).to(self.device) for slot in self.config.dst_slot_list}

        self.global_eval_stats = copy.deepcopy(TEMPLATE_STATE)
        for d in self.global_eval_stats:
            for s in self.global_eval_stats[d]:
                self.global_eval_stats[d][s] = {'TP': 0, 'FP': 0, 'FN': 0}
        self.global_diag_cnt = -3
        self.global_turn_cnt = -1

        self.load_weights()
    
    def load_weights(self):
        self.tokenizer = self.tokenizer_class.from_pretrained(self.model_name, local_files_only=True) # TODO: do_lower_case=args.do_lower_case ? # TODO: parameterize
        self.model = self.model_class.from_pretrained(self.model_path, config=self.config, local_files_only=True) # TODO: parameterize
        self.model.to(self.device)
        self.model.eval()
        self.nlu = BERTNLU(model_file=self.nlu_path) # This is used for internal evaluation
        self.nlg_usr = TemplateNLG(is_user=True)
        self.nlg_sys = TemplateNLG(is_user=False)
        
    def init_session(self):
        self.state = default_state() # Initialise as empty state
        self.state['belief_state'] = copy.deepcopy(TEMPLATE_STATE)
        self.nlg_history = []
        self.ds_aux = {slot: torch.tensor([0]).to(self.device) for slot in self.config.dst_slot_list}
        self.gt_belief_state = copy.deepcopy(TEMPLATE_STATE)
        self.global_diag_cnt += 1
        self.global_turn_cnt = -1

    def update_gt_belief_state(self, user_act):
        for intent, domain, slot, value in user_act:
            if domain == 'police':
                continue
            if intent == 'inform':
                if slot == 'none' or slot == '':
                    continue
                domain_dic = self.gt_belief_state[domain]
                if slot in domain_dic:
                    #nvalue = normalize_value(self.value_dict, domain, slot, value)
                    self.gt_belief_state[domain][slot] = value # nvalue
                #elif slot != 'none' or slot != '':
                #    raise Exception('Unknown slot name <{}> with value <{}> of domain <{}>'.format(slot, value, domain))

    # TODO: receive semantic, convert semantic -> text -> semantic for sanity check
    # For TripPy: receive semantic, convert semantic -> text (with context) as input to DST
    # - allows for accuracy estimates
    # - allows isolating inform prediction from request prediction (as can be taken from input for sanity check)
    def update(self, user_act=''):
        def normalize_values(text):
            text_to_num = {"zero": "0", "one": "1", "me": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7"}
            #text = re.sub("^(\d{2}) : (\d{2})$", r"\1:\2", text) # Times
            #text = re.sub(" ?' ?s", "s", text) # Genitive
            text = re.sub("\s*(\W)\s*", r"\1" , text) # Re-attach special characters
            text = re.sub("s'([^s])", r"s' \1", text) # Add space after plural genitive apostrophe
            if text in text_to_num:
                text = text_to_num[text]
            return text

        prev_state = self.state

        if not self.no_eval:
            print("-" * 40)

        #nlg_history = []
        ##for h in prev_state['history'][-2:]: # TODO: make this an option?
        #for h in prev_state['history']:
        #    nlg_history.append([h[0], self.get_text(h[1], is_user=(h[0]=='user'))])
        ## Special case: at the beginning of the dialog the history might be empty (depending on policy)
        #if len(nlg_history) == 0:
        #    nlg_history.append(['sys', self.get_text(prev_state['system_action'], is_user=False)])
        #    nlg_history.append(['user', self.get_text(prev_state['user_action'], is_user=True)])
        if self.no_history:
            self.nlg_history = []
        self.nlg_history.append(['sys', self.get_text(prev_state['system_action'], is_user=False, normalize=True)])
        self.nlg_history.append(['user', self.get_text(prev_state['user_action'], is_user=True, normalize=True)])
        self.global_turn_cnt += 1
        if not self.no_eval:
            self.print_dialog(self.nlg_history)

        # --- Get inform memory and auxiliary features ---

        # If system_action is plain text, get acts using NLU
        if isinstance(prev_state['user_action'], str):
            u_acts, s_acts = self.get_acts()
        elif isinstance(prev_state['user_action'], list):
            u_acts = user_act # same as prev_state['user_action']
            s_acts = prev_state['system_action']
        else:
            raise Exception('Unknown format for user action:', prev_state['user_action'])
        inform_aux, inform_mem = self.get_inform_aux(s_acts)
        if not self.no_eval:
            self.print_inform_memory(inform_mem)

        # --- Tokenize dialogue context and feed DST model ---

        ##features = self.get_features(self.state['history'], ds_aux=self.ds_aux, inform_aux=inform_aux)
        used_ds_aux = None if not self.config.dst_class_aux_feats_ds else self.ds_aux
        used_inform_aux = None if not self.config.dst_class_aux_feats_inform else inform_aux
        features = self.get_features(self.nlg_history, ds_aux=used_ds_aux, inform_aux=used_inform_aux)
        pred_states, pred_classes, cls_representation = self.predict(features, inform_mem)

        # --- Update ConvLab-style dialogue state ---

        new_belief_state = copy.deepcopy(prev_state['belief_state'])
        user_acts = []
        for state, value in pred_states.items():
            value = normalize_values(value)
            if value == 'none':
                continue
            domain, slot = state.split('-', 1)
            # Value normalization # TODO: according to trippy rules?
            if domain == 'hotel' and slot == 'type':
                value = "hotel" if value == "yes" else "guesthouse"
            if not self.no_normalize_value:
                value = normalize_value(self.value_dict, domain, slot, value)
            slot = SLOT_MAP_TRIPPY_TO_UDF[domain].get(slot, slot)
            if slot in new_belief_state[domain]:
                new_belief_state[domain][slot] = value
                user_acts.append(['inform', domain, SLOT_MAP_TRIPPY_TO_UDF[domain].get(slot, slot), value])
            else:
                raise Exception('Unknown slot name <{}> with value <{}> of domain <{}>'.format(slot, value, domain))

        self.update_gt_belief_state(u_acts) # For evaluation

        # BELIEF STATE UPDATE
        new_state = copy.deepcopy(dict(prev_state))
        new_state['belief_state'] = new_belief_state # TripPy
        if self.gt_ds:
            new_state['belief_state'] = self.gt_belief_state # Rule

        state_updates = {}
        for cl in pred_classes:
            cl_d, cl_s = cl.split('-')
            # Some reformatting for the evaluation further down
            if cl_d not in state_updates:
                state_updates[cl_d] = {}
            state_updates[cl_d][SLOT_MAP_TRIPPY_TO_UDF[cl_d].get(cl_s, cl_s)] = pred_classes[cl]
            # We care only about the requestable slots here
            if self.config.dst_class_types[pred_classes[cl]] != 'request':
                continue
            if cl_d != 'general' and cl_s == 'none':
                user_acts.append(['inform', cl_d, '', ''])
            elif cl_d == 'general':
                user_acts.append([SLOT_MAP_TRIPPY_TO_UDF[cl_d].get(cl_s, cl_s), 'general', '', ''])
                #user_acts.append(['bye', 'general', '', '']) # Map "thank" to "bye"? Mind "hello" as well!
            elif not self.gt_request_acts:
                user_acts.append(['request', cl_d, SLOT_MAP_TRIPPY_TO_UDF[cl_d].get(cl_s, cl_s), ''])

        # TODO: For debugging -> doesn't make a difference
        #for e in user_act:
        #    nlu_a, nlu_d, nlu_s, nlu_v = e
        #    nlu_a = nlu_a.lower()
        #    nlu_d = nlu_d.lower()
        #    nlu_s = nlu_s.lower()
        #    nlu_v = nlu_v.lower()
        #    # Mostly requestables
        #    if nlu_a == 'inform' and nlu_d == 'train' and nlu_s == 'notbook':
        #        user_acts.append([nlu_a, nlu_d, 'NotBook', 'none'])

        # TODO: fix # TODO: still needed?
        if 0:
            domain = ''
            is_inform = False
            is_request = False
            is_notbook = False
            for act in user_act:
                _, _, slot, _ = act
                if slot == "NotBook":
                    is_notbook = True
            for act in user_acts:
                intent, domain, slot, value = act
                if intent == 'inform':
                    is_inform = True
                if intent == 'request':
                    is_request = True
            if is_inform and not is_request and not is_notbook and domain != '' and domain != "general":
                user_acts = [['inform', domain, '', '']] + user_acts

        # USER ACTS UPDATE
        new_state['user_action'] = user_acts # TripPy
        # ONLY FOR DEBUGGING
        if self.gt_user_acts:
            new_state['user_action'] = u_acts # Rule
        elif self.gt_request_acts:
            for e in u_acts:
                ea, _, _, _ = e
                if ea == 'request':
                    user_acts.append(e)

        if not self.no_eval:
            self.eval_user_acts(u_acts, user_acts)
            self.eval_dialog_state(state_updates, new_belief_state)

        #new_state['cls_representation'] = cls_representation # TODO: needed by Nunu?

        self.state = new_state

        # Print eval statistics
        if self.state['terminated'] and not self.no_eval:
            print("Booked:", self.state['booked'])
            self.eval_print_stats()
            print("=" * 10, "End of the dialogue", "=" * 10)
            #self.ds_aux = {slot: torch.tensor([0]).to(self.device) for slot in self.config.dst_slot_list}
        #else:
        self.ds_aux = self.update_ds_aux(self.state['belief_state'], pred_states)
        #print("ds:", [self.ds_aux[s][0].item() for s in self.ds_aux])

        return self.state
    
    def predict(self, features, inform_mem):
        #aaa_time = time.time()
        with torch.no_grad():
            outputs = self.model(input_ids=features['input_ids'],
                                 input_mask=features['attention_mask'],
                                 inform_slot_id=features['inform_slot_id'],
                                 diag_state=features['diag_state'])
        #bbb_time = time.time()

        input_tokens = self.tokenizer.convert_ids_to_tokens(features['input_ids'][0]) # unmasked!

        #total_loss = outputs[0]
        #per_slot_per_example_loss = outputs[1]
        per_slot_class_logits = outputs[2]
        per_slot_start_logits = outputs[3]
        per_slot_end_logits = outputs[4]
        per_slot_refer_logits = outputs[5]

        cls_representation = outputs[6]
            
        # TODO: maybe add assert to check that batch=1
        
        predictions = {slot: 'none' for slot in self.config.dst_slot_list}
        class_predictions = {slot: 0 for slot in self.config.dst_slot_list}

        for slot in self.config.dst_slot_list:
            class_logits = per_slot_class_logits[slot][0]
            start_logits = per_slot_start_logits[slot][0]
            end_logits = per_slot_end_logits[slot][0]
            refer_logits = per_slot_refer_logits[slot][0]

            class_prediction = int(class_logits.argmax())
            start_prediction = int(start_logits.argmax())
            end_prediction = int(end_logits.argmax())
            refer_prediction = int(refer_logits.argmax())

            if class_prediction == self.config.dst_class_types.index('dontcare'):
                predictions[slot] = 'dontcare'
            elif class_prediction == self.config.dst_class_types.index('copy_value'):
                predictions[slot] = ' '.join(input_tokens[start_prediction:end_prediction + 1])
                predictions[slot] = re.sub("(^| )##", "", predictions[slot])
                if "\u0120" in predictions[slot]:
                    predictions[slot] = re.sub(" ", "", predictions[slot])
                    predictions[slot] = re.sub("\u0120", " ", predictions[slot])
                    predictions[slot] = predictions[slot].strip()
            elif 'true' in self.config.dst_class_types and class_prediction == self.config.dst_class_types.index('true'):
                predictions[slot] = "yes" # 'true'
            elif 'false' in self.config.dst_class_types and class_prediction == self.config.dst_class_types.index('false'):
                predictions[slot] = "no" # 'false'
            elif class_prediction == self.config.dst_class_types.index('inform'):
                #print("INFORM:", slot, ",", predictions[slot], "->", inform_mem[slot])
                predictions[slot] = inform_mem[slot]
            # Referral case is handled below

        # Referral case. All other slot values need to be seen first in order
        # to be able to do this correctly.
        for slot in self.config.dst_slot_list:
            class_logits = per_slot_class_logits[slot][0]
            refer_logits = per_slot_refer_logits[slot][0]

            class_prediction = int(class_logits.argmax())
            refer_prediction = int(refer_logits.argmax())

            if 'refer' in self.config.dst_class_types and class_prediction == self.config.dst_class_types.index('refer'):
                # Only slots that have been mentioned before can be referred to.
                # First try to resolve a reference within the same turn. (One can think of a situation
                # where one slot is referred to in the same utterance. This phenomenon is however
                # currently not properly covered in the training data label generation process)
                # Then try to resolve a reference given the current dialogue state.
                predictions[slot] = predictions[self.config.dst_slot_list[refer_prediction - 1]]
                if predictions[slot] == 'none':
                    referred_slot = self.config.dst_slot_list[refer_prediction - 1]
                    referred_slot_d, referred_slot_s = referred_slot.split('-')
                    referred_slot_s = SLOT_MAP_TRIPPY_TO_UDF[referred_slot_d].get(referred_slot_s, referred_slot_s)
                    if self.state['belief_state'][referred_slot_d][referred_slot_s] != '':
                        predictions[slot] = self.state['belief_state'][referred_slot_d][referred_slot_s]
                if predictions[slot] == 'none':
                    ref_slot = self.config.dst_slot_list[refer_prediction - 1]
                    if ref_slot == 'hotel-name':
                        predictions[slot] = 'the hotel'
                    elif ref_slot == 'restaurant-name':
                        predictions[slot] = 'the restaurant'
                    elif ref_slot == 'attraction-name':
                        predictions[slot] = 'the attraction'
                    elif ref_slot == 'hotel-area':
                        predictions[slot] = 'same area as the hotel'
                    elif ref_slot == 'restaurant-area':
                        predictions[slot] = 'same area as the restaurant'
                    elif ref_slot == 'attraction-area':
                        predictions[slot] = 'same area as the attraction'
                    elif ref_slot == 'hotel-pricerange':
                        predictions[slot] = 'in the same price range as the hotel'
                    elif ref_slot == 'restaurant-pricerange':
                        predictions[slot] = 'in the same price range as the restaurant'

            class_predictions[slot] = class_prediction
            #if class_prediction > 0:
            #    print("  ", slot, "->", class_prediction, ",", predictions[slot])
        #ccc_time = time.time()
        #print("TIME:", bbb_time - aaa_time, ccc_time - bbb_time)

        return predictions, class_predictions, cls_representation

    def get_features(self, context, ds_aux=None, inform_aux=None):
        assert(self.model_type == "roberta") # TODO: generalize to other BERT-like models
        input_tokens = ['<s>']
        e_itr = 0
        for e_itr, e in enumerate(reversed(context)):
            #input_tokens.append(e[1].lower() if e[1] != 'null' else ' ') # TODO: normalise text
            input_tokens.append(e[1] if e[1] != 'null' else ' ') # TODO: normalise text
            if e_itr < 2:
                input_tokens.append('</s> </s>')
        if e_itr == 0:
            input_tokens.append('</s> </s>')
        input_tokens.append('</s>')
        input_tokens = ' '.join(input_tokens)

        # TODO: delex sys utt somehow, or refrain from using delex for sys utts?
        features = self.tokenizer.encode_plus(input_tokens, add_special_tokens=False, max_length=self.config.dst_max_seq_length)

        input_ids = torch.tensor(features['input_ids']).reshape(1,-1).to(self.device)
        attention_mask = torch.tensor(features['attention_mask']).reshape(1,-1).to(self.device)
        features = {'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'inform_slot_id': inform_aux,
                    'diag_state': ds_aux}

        return features

    def update_ds_aux(self, state, pred_states, terminated=False):
        ds_aux = copy.deepcopy(self.ds_aux) # TODO: deepcopy necessary? just update class variable?
        for slot in self.config.dst_slot_list:
            d, s = slot.split('-')
            if d in state and s in state[d]:
                ds_aux[slot][0] = int(state[d][SLOT_MAP_TRIPPY_TO_UDF[d].get(s, s)] != '')
            else:
                # Requestable slots are not found in the DS
                ds_aux[slot][0] = int(pred_states[slot] != 'none')
        return ds_aux

    # TODO: consider "booked" values?
    def get_inform_aux(self, state):
        inform_aux = {slot: torch.tensor([0]).to(self.device) for slot in self.config.dst_slot_list}
        inform_mem = {slot: 'none' for slot in self.config.dst_slot_list}
        for e in state:
            #print(e)
            #pdb.set_trace()
            a, d, s, v = e
            if a in ['inform', 'recommend', 'select', 'book', 'offerbook']:
                #ds_d = d.lower()
                #if s in REF_SYS_DA[d]:
                #    ds_s = REF_SYS_DA[d][s]
                #elif s in REF_SYS_DA['Booking']:
                #    ds_s = "book_" + REF_SYS_DA['Booking'][s]
                #else:
                #    ds_s = s.lower()
                #    #raise Exception('Slot <{}> of domain <{}> unknown'.format(s, d))
                slot = "%s-%s" % (d, s)
                if slot in inform_aux:
                    inform_aux[slot][0] = 1
                    inform_mem[slot] = v
        return inform_aux, inform_mem

    def get_acts(self):
        context = self.state['history']
        if context[-1][0] != 'user':
            raise Exception("Wrong order of utterances, check your input.")
        system_act = context[-2][-1]
        user_act = context[-1][-1]
        system_context = [t for s,t in context[:-2]]
        user_context = [t for s,t in context[:-1]]

        #print("  SYS:", system_act, system_context)
        system_acts = self.nlu.predict(system_act, context=system_context)

        #print("  USR:", user_act, user_context)
        user_acts = self.nlu.predict(user_act, context=user_context)
        
        return user_acts, system_acts

    def get_text(self, act, is_user=False, normalize=False):
        if act == 'null':
            return 'null'
        if not isinstance(act, list):
            result = act
        elif is_user:
            result = self.nlg_usr.generate(act)
        else:
            result = self.nlg_sys.generate(act)
        if normalize:
            return self.normalize_text(result)
        else:
            return result

    def normalize_text(self, text):
        norm_text = text.lower()
        #norm_text = re.sub("n't", " not", norm_text) # Does not make much of a difference
        #norm_text = re.sub("ca not", "cannot", norm_text)
        norm_text = ' '.join([tok for tok in map(str.strip, re.split("(\W+)", norm_text)) if len(tok) > 0])
        return norm_text


# if __name__ == "__main__":
#     tracker = TRIPPY(model_type='roberta', model_path='/path/to/model',
#                         nlu_path='/path/to/nlu')
#     tracker.init_session()
#     state = tracker.update('hey. I need a cheap restaurant.')
#     tracker.state['history'].append(['usr', 'hey. I need a cheap restaurant.'])
#     tracker.state['history'].append(['sys', 'There are many cheap places, which food do you like?'])
#     state = tracker.update('If you have something Asian that would be great.')
#     tracker.state['history'].append(['usr', 'If you have something Asian that would be great.'])
#     tracker.state['history'].append(['sys', 'The Golden Wok is a nice cheap chinese restaurant.'])
#     state = tracker.update('Great. Where are they located?')
#     tracker.state['history'].append(['usr', 'Great. Where are they located?'])
#     print(tracker.state)
