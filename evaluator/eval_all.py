import json
from pathlib import Path
from ngram_metrics.bleu.bleu import Bleu
from ngram_metrics.cider.cider import Cider
from ngram_metrics.meteor.meteor import Meteor
from ngram_metrics.rouge.rouge import Rouge
import pandas as pd
import re
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim

import evaluate



def clean_answer(data):
    data = data.lower()
    data = re.sub('[ ]+$' ,'', data)
    data = re.sub('^[ ]+' ,'', data)
    data = re.sub(' {2,}', ' ', data)

    data = re.sub('\.[ ]{2,}', '. ', data)
    data = re.sub('[^a-zA-Z0-9,\'\s\-:]+', '', data)
    data = re.sub('ç' ,'c', data)
    data = re.sub('’' ,'\'', data)
    data = re.sub(r'\bletf\b' ,'left', data)
    data = re.sub(r'\blet\b' ,'left', data)
    data = re.sub(r'\btehre\b' ,'there', data)
    data = re.sub(r'\brigth\b' ,'right', data)
    data = re.sub(r'\brght\b' ,'right', data)
    data = re.sub(r'\bbehine\b', 'behind', data)
    data = re.sub(r'\btv\b' ,'TV', data)
    data = re.sub(r'\bchai\b' ,'chair', data)
    data = re.sub(r'\bwasing\b' ,'washing', data)
    data = re.sub(r'\bwaslked\b' ,'walked', data)
    data = re.sub(r'\boclock\b' ,'o\'clock', data)
    data = re.sub(r'\bo\'[ ]+clock\b' ,'o\'clock', data)

    # digit to word, only for answer
    data = re.sub(r'\b0\b', 'zero', data)
    data = re.sub(r'\bnone\b', 'zero', data)
    data = re.sub(r'\b1\b', 'one', data)
    data = re.sub(r'\b2\b', 'two', data)
    data = re.sub(r'\b3\b', 'three', data)
    data = re.sub(r'\b4\b', 'four', data)
    data = re.sub(r'\b5\b', 'five', data)
    data = re.sub(r'\b6\b', 'six', data)
    data = re.sub(r'\b7\b', 'seven', data)
    data = re.sub(r'\b8\b', 'eight', data)
    data = re.sub(r'\b9\b', 'nine', data)
    data = re.sub(r'\b10\b', 'ten', data)
    data = re.sub(r'\b11\b', 'eleven', data)
    data = re.sub(r'\b12\b', 'twelve', data)
    data = re.sub(r'\b13\b', 'thirteen', data)
    data = re.sub(r'\b14\b', 'fourteen', data)
    data = re.sub(r'\b15\b', 'fifteen', data)
    data = re.sub(r'\b16\b', 'sixteen', data)
    data = re.sub(r'\b17\b', 'seventeen', data)
    data = re.sub(r'\b18\b', 'eighteen', data)
    data = re.sub(r'\b19\b', 'nineteen', data)
    data = re.sub(r'\b20\b', 'twenty', data)
    data = re.sub(r'\b23\b', 'twenty-three', data)

    # misc
    # no1, mat2, etc
    data = re.sub(r'\b([a-zA-Z]+)([0-9])\b' ,r'\g<1>', data)
    data = re.sub(r'\ba\b ([a-zA-Z]+)' ,r'\g<1>', data)
    data = re.sub(r'\ban\b ([a-zA-Z]+)' ,r'\g<1>', data)
    data = re.sub(r'\bthe\b ([a-zA-Z]+)' ,r'\g<1>', data)

    data = re.sub(r'\bbackwards\b', 'backward', data)

    return data

# Assuming the ScanQAEvaluator class is already defined as provided
# Now, let’s write the function to load the JSON and evaluate the predictions

class ScanQAEvaluator():
    def __init__(self):

        self.cider_scorer = Cider()
        self.bleu_scorer = Bleu()
        self.meteor_scorer = Meteor()
        self.rouge_scorer = Rouge()

        self.reset()

    def reset(self):
        self.eval_dict = {
            'target_metric': [], 'em': [], 'em_refined': [],
            'cider': 0, 'bleu': 0, 'meteor': 0, 'rouge': 0,
        }
        self.total_count = 0
        self.save_results = []
        self.gt_sentence_mp = []
        self.pred_sentence_mp = []

    def answer_match(self, pred, gts):
        # return EM and refined EM
        for gt in gts:
            if pred == gt:
                return 1, 1
            elif ''.join(pred.split()) in ''.join(gt.split()):
                return 0, 1
            elif ''.join(gt.split()) in ''.join(pred.split()):
                return 0, 1
        return 0, 0

    def batch_metrics(self, data_dict):
        metrics = {}
        em = 0
        em_refined = 0
        for answer_pred, answer_gts in zip(data_dict['output_txt'], data_dict['output_gt']):
            answer_pred = clean_answer(answer_pred)
            answer_gts = [clean_answer(gt) for gt in answer_gts]
            em_flag, em_refined_flag = self.answer_match(pred=answer_pred, gts=answer_gts)
            em += em_flag
            em_refined += em_refined_flag

            self.pred_sentence_mp.append([answer_pred])
            self.gt_sentence_mp.append(answer_gts)

        batch_size = len(data_dict['output_gt'])
        metrics['total_count'] = batch_size
        metrics['em'] = em / batch_size
        metrics['em_refined'] = em_refined / batch_size
        metrics['target_metric'] = metrics['em_refined']
        return metrics

    def update(self, data_dict):
        metrics = self.batch_metrics(data_dict)
        batch_size = metrics['total_count']
        self.total_count += batch_size
        for i in range(batch_size):
            self.save_results.append({
                # vision
                # language
                'response_gt': data_dict['output_gt'][i],
                'response_pred': data_dict['output_txt'][i],
            })

        for key in self.eval_dict.keys():
            if key not in ['cider', 'bleu', 'meteor', 'rouge']:
                self.eval_dict[key].append(metrics[key] * batch_size)

    def record(self, split, is_main_process):
        # ngram metrics
        self.gt_sentence_mp = {k: v for k, v in enumerate(self.gt_sentence_mp)}
        self.pred_sentence_mp = {k: v for k, v in enumerate(self.pred_sentence_mp)}

        self.eval_dict['cider'] = self.cider_scorer.compute_score(self.gt_sentence_mp, self.pred_sentence_mp)[0]
        self.eval_dict['bleu'] = self.bleu_scorer.compute_score(self.gt_sentence_mp, self.pred_sentence_mp)[0][-1]
        self.eval_dict['meteor'] = self.meteor_scorer.compute_score(self.gt_sentence_mp, self.pred_sentence_mp)[0]
        self.eval_dict['rouge'] = self.rouge_scorer.compute_score(self.gt_sentence_mp, self.pred_sentence_mp)[0]

        # others
        for k, v in self.eval_dict.items():
            if k not in ['cider', 'bleu', 'meteor', 'rouge']:
                self.eval_dict[k] = sum(v) / self.total_count

        # if self.eval_dict['target_metric'] > self.best_result:
        #     is_best = True
        #     self.best_result = self.eval_dict['target_metric']
        # else:
        #     is_best = False

        # if (is_best or split == 'test') and is_main_process:
        #     with open(str(self.save_dir / 'results.json'), 'w') as f:
        #         json.dump(self.save_results, f, indent=2)

        return True, self.eval_dict


class CaptionEvaluator():
    def __init__(self, task_name):
        self.task_name=task_name
        self.cider_scorer = Cider()
        self.bleu_scorer = Bleu()
        self.meteor_scorer = Meteor()
        self.rouge_scorer = Rouge()

        self.best_result = -np.inf

        
        self.sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='../all-MiniLM-L6-v2')


        self.reset()

    def reset(self):
        self.eval_dict = {
            'target_metric': [], 'sentence_sim': [],
            'cider': 0, 'bleu': 0, 'meteor': 0, 'rouge': 0,
        }
        self.total_count = 0
        self.save_results = []
        self.init_corpus()

    def init_corpus(self):
        if self.task_name.lower() == 'scan2cap':
            with open("../annotations/instruction/scan2cap/scanrefer_corpus.json", 'r') as f:
                gt_sentence_mp = json.load(f)
                self.gt_sentence_mp={}
                for k, v in gt_sentence_mp.items():
                    self.gt_sentence_mp[k.replace(" ", "_")]=v
            self.pred_sentence_mp = {}
        else:
            # init with list, finally convert to dict
            self.gt_sentence_mp = []
            self.pred_sentence_mp = []

    # def batch_metrics(self, data_dict):
    #     metrics = {}
    #     output_gt = data_dict['output_gt']
    #     output_pred = data_dict['output_txt']
    #     batch_size = len(output_gt)

    #     # consider IoU-based caption metrics
    #     if 'iou_flag' in data_dict:
    #         iou_flags = data_dict['iou_flag']
    #     else:
    #         iou_flags = [True] * batch_size

    #     if self.task_name.lower() == 'scan2cap':
    #         for i in range(batch_size):
    #             corpus_key = data_dict['corpus_key'][i]
    #             if iou_flags[i]:
    #                 self.pred_sentence_mp[corpus_key] = [('sos ' + output_pred[i] + ' eos').replace('. ', ' . ')]
    #             else:
    #                 output_pred[i] = ""
    #                 self.pred_sentence_mp[corpus_key] = ["sos eos"]
    #     else:
    #         for i in range(batch_size):
    #             if iou_flags[i]:
    #                 self.pred_sentence_mp.append([output_pred[i]])
    #             else:
    #                 output_pred[i] = ""
    #                 self.pred_sentence_mp.append([""])
    #             self.gt_sentence_mp.append([output_gt[i]])

    #     # compute sentence similarity
    #     embed_pred = self.sentence_model.encode(output_pred, convert_to_tensor=True)
    #     embed_gt = self.sentence_model.encode(output_gt, convert_to_tensor=True)
    #     sims = pytorch_cos_sim(embed_pred, embed_gt).diag()

    #     metrics['total_count'] = batch_size
    #     metrics['sentence_sim'] = sims.mean().item()
    #     metrics['target_metric'] = metrics['sentence_sim']
    #     return metrics

    def batch_metrics(self, data_dict):
        metrics = {}
        output_gt = data_dict['output_gt']
        output_pred = data_dict['output_txt']
        batch_size = len(output_gt)
    
        # Flatten and sanitize inputs to ensure they are strings
        output_gt = [str(x) for x in output_gt]
        output_pred = [str(x) for x in output_pred]
    
        # consider IoU-based caption metrics
        if 'iou_flag' in data_dict:
            iou_flags = data_dict['iou_flag']
        else:
            iou_flags = [True] * batch_size
    
        if self.task_name.lower() == 'scan2cap':
            for i in range(batch_size):
                corpus_key = data_dict['corpus_key'][i]
                if iou_flags[i]:
                    self.pred_sentence_mp[corpus_key] = [('sos ' + output_pred[i] + ' eos').replace('. ', ' . ')]
                else:
                    output_pred[i] = ""
                    self.pred_sentence_mp[corpus_key] = ["sos eos"]
        else:
            for i in range(batch_size):
                if iou_flags[i]:
                    self.pred_sentence_mp.append([output_pred[i]])
                else:
                    output_pred[i] = ""
                    self.pred_sentence_mp.append([""])
                self.gt_sentence_mp.append([output_gt[i]])
    
        # compute sentence similarity
        embed_pred = self.sentence_model.encode(output_pred, convert_to_tensor=True)
        embed_gt = self.sentence_model.encode(output_gt, convert_to_tensor=True)
        sims = pytorch_cos_sim(embed_pred, embed_gt).diag()
    
        metrics['total_count'] = batch_size
        metrics['sentence_sim'] = sims.mean().item()
        metrics['target_metric'] = metrics['sentence_sim']
        return metrics


    def update(self, data_dict):
        metrics = self.batch_metrics(data_dict)
        batch_size = metrics['total_count']
        self.total_count += batch_size

        for i in range(batch_size):
            save_dict = {
                    # vision
                    # language
                    'response_gt': data_dict['output_gt'][i],
                    'response_pred': data_dict['output_txt'][i],
            }
            
            if 'iou_flag' in data_dict:
                save_dict['iou_flag'] = data_dict['iou_flag'][i].item()
            self.save_results.append(save_dict)

        for key in self.eval_dict.keys():
            if key not in ['cider', 'bleu', 'meteor', 'rouge']:
                self.eval_dict[key].append(metrics[key] * batch_size)

    def record(self, split, is_main_process):
        # ngram metrics
        if self.task_name.lower() == 'scan2cap':
            # align gt_sentence_mp to pred_sentence_mp for partial evaluation
            self.gt_sentence_mp = {k: self.gt_sentence_mp[k] for k in self.pred_sentence_mp.keys()}
        else:
            self.gt_sentence_mp = {k: v for k, v in enumerate(self.gt_sentence_mp)}
            self.pred_sentence_mp = {k: v for k, v in enumerate(self.pred_sentence_mp)}

        self.eval_dict['cider'] = self.cider_scorer.compute_score(self.gt_sentence_mp, self.pred_sentence_mp)[0]
        self.eval_dict['bleu'] = self.bleu_scorer.compute_score(self.gt_sentence_mp, self.pred_sentence_mp)[0][-1]
        
        ms = []
        for k in self.gt_sentence_mp:
            gt = {0: self.gt_sentence_mp[k]}       # reference must be a list of strings
            pred = {0: self.pred_sentence_mp[k]}   # prediction must be a string
            score=self.meteor_scorer.compute_score(gt, pred)
            ms.append(score)
        
        self.eval_dict['meteor'] = sum(ms) / len(ms)
        arr = np.array(ms)

        print("Mean:", np.mean(arr))
        print("Median:", np.median(arr))
        print("Min:", np.min(arr))
        print("Max:", np.max(arr))
        print("Quantiles:")
        for q in [0, 0.25, 0.5, 0.75, 1.0]:
            print(f"  {int(q*100)}%: {np.quantile(arr, q)}")
        self.eval_dict['rouge'] = self.rouge_scorer.compute_score(self.gt_sentence_mp, self.pred_sentence_mp)[0]

        # others
        for k, v in self.eval_dict.items():
            if k not in ['cider', 'bleu', 'meteor', 'rouge']:
                self.eval_dict[k] = sum(v) / self.total_count

        

        return True, self.eval_dict

class SQA3DEvaluator(ScanQAEvaluator):
    def get_sqa_question_type(self, question):
        question = question.lstrip()
        if question[:4].lower() == 'what':
            return 0
        elif question[:2].lower() == 'is':
            return 1
        elif question[:3].lower() == 'how':
            return 2
        elif question[:3].lower() == 'can':
            return 3
        elif question[:5].lower() == 'which':
            return 4
        else:
            return 5   # others

    
    def reset(self):
        self.eval_dict = {
            'target_metric': [], 'em_overall': [], 'em_refined_overall': [],
            'em_type0': [], 'em_refined_type0': [], 'em_type1': [], 'em_refined_type1': [],
            'em_type2': [], 'em_refined_type2': [], 'em_type3': [], 'em_refined_type3': [],
            'em_type4': [], 'em_refined_type4': [], 'em_type5': [], 'em_refined_type5': [],
            'cider_overall': 0, 'bleu_overall': 0, 'meteor_overall': 0, 'rouge_overall': 0,
        }
        self.total_count = 0
        self.type_count = {0: 1e-10, 1: 1e-10, 2: 1e-10, 3: 1e-10, 4: 1e-10, 5: 1e-10}
        self.save_results = []
        self.gt_sentence_mp = []
        self.pred_sentence_mp = []

    def batch_metrics(self, data_dict):
        metrics = {
            'type0_count': 1e-10, 'type1_count': 1e-10, 'type2_count': 1e-10,
            'type3_count': 1e-10, 'type4_count': 1e-10, 'type5_count': 1e-10,
        }

        em_overall = 0
        em_refined_overall = 0
        em_type = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        em_refined_type = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

        for answer_pred, answer_gts, instruction in zip(
            data_dict['output_txt'], data_dict['output_gt'], data_dict["instruction"]
        ):
            answer_pred = clean_answer(answer_pred)
            answer_gts = [clean_answer(gt) for gt in answer_gts]
            em_flag, em_refined_flag = self.answer_match(pred=answer_pred, gts=answer_gts)
            em_overall += em_flag
            em_refined_overall += em_refined_flag

            sqa_type = int(self.get_sqa_question_type(instruction.split("USER:")[1]))   # 0-dim tensor to int
            em_type[sqa_type] += em_flag
            em_refined_type[sqa_type] += em_refined_flag
            metrics[f'type{sqa_type}_count'] += 1

            self.pred_sentence_mp.append([answer_pred])
            self.gt_sentence_mp.append(answer_gts)

        batch_size = len(data_dict['output_gt'])
        metrics['total_count'] = batch_size
        metrics['em_overall'] = em_overall / batch_size
        metrics['em_refined_overall'] = em_refined_overall / batch_size
        for key in em_type.keys():
            metrics[f'em_type{key}'] = em_type[key] / metrics[f'type{key}_count']
            metrics[f'em_refined_type{key}'] = em_refined_type[key] / metrics[f'type{key}_count']

        metrics['target_metric'] = metrics['em_refined_overall']
        return metrics

    def update(self, data_dict):
        metrics = self.batch_metrics(data_dict)
        batch_size = metrics['total_count']
        self.total_count += batch_size
        for key in metrics.keys():
            if 'type' in key and 'count' in key:
                # type{x}_count
                self.type_count[int(key[4])] += metrics[key]
       
        for i in range(batch_size):
            self.save_results.append({
                # vision
               
                'response_gt': data_dict['output_gt'][i],
                'response_pred': data_dict['output_txt'][i],
            })

        # save eval dict
        for key in self.eval_dict.keys():
            if key in ['cider_overall', 'bleu_overall', 'meteor_overall', 'rouge_overall']:
                continue
            if 'type' in key:
                self.eval_dict[key].append(metrics[key] * metrics[f'type{key[-1]}_count'])
            else:
                self.eval_dict[key].append(metrics[key] * batch_size)

    def record(self, split, is_main_process):
        # ngram metrics
        self.gt_sentence_mp = {k: v for k, v in enumerate(self.gt_sentence_mp)}
        self.pred_sentence_mp = {k: v for k, v in enumerate(self.pred_sentence_mp)}

        self.eval_dict['cider_overall'] = self.cider_scorer.compute_score(self.gt_sentence_mp, self.pred_sentence_mp)[0]
        self.eval_dict['bleu_overall'] = self.bleu_scorer.compute_score(self.gt_sentence_mp, self.pred_sentence_mp)[0][-1]
        self.eval_dict['meteor_overall'] = self.meteor_scorer.compute_score(self.gt_sentence_mp, self.pred_sentence_mp)[0]
        self.eval_dict['rouge_overall'] = self.rouge_scorer.compute_score(self.gt_sentence_mp, self.pred_sentence_mp)[0]

        # others
        for k, v in self.eval_dict.items():
            if k in ['cider_overall', 'bleu_overall', 'meteor_overall', 'rouge_overall']:
                continue
            if 'type' in k:
                self.eval_dict[k] = sum(v) / self.type_count[int(k[-1])]
            else:
                self.eval_dict[k] = sum(v) / self.total_count

       


        return True, self.eval_dict



def load_json_data(file_path):
    """Load JSON data from a given file path."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def prepare_data_for_evaluation(df):
    """Prepare the data format for evaluation."""
    data_dict = {
        'output_txt': [],
        'output_gt': [],
    }

    # Assuming the structure of the JSON matches the evaluator's needs
    data_dict['output_txt']=list(df['response_pred'])
    data_dict['output_gt']=list(df['response_gt'])

    return data_dict

def prepare_data_for_evaluation_sqa3d(df):
    """Prepare the data format for evaluation."""
    data_dict = {
        'output_txt': [],
        'output_gt': [],
        'instruction': []
    }

    # Assuming the structure of the JSON matches the evaluator's needs
    data_dict['output_txt']=list(df['response_pred'])
    data_dict['output_gt']=list(df['response_gt'])
    data_dict['instruction']=list(df['instruction'])

    return data_dict

def prepare_data_for_evaluation_scan2cap(df):
    """Prepare the data format for evaluation."""
    data_dict = {
        'output_txt': [],
        'output_gt': [],
        'corpus_key': []
    }

    # Assuming the structure of the JSON matches the evaluator's needs
    #print(f"Before filtering {len(df)}")
    # df = df[
    #         #~df['response_gt'].str.contains("right", case=False, na=False) &
    #         ~df['response_gt'].str.contains("chair", case=False, na=False)
    #     ]
    #df['response_pred'] = df['response_pred'].str.replace(r'\b(this is|it is)\b', '', case=False, regex=True).str.strip()

    
    #print(f"Before filtering {len(df)}")
    data_dict['output_txt']=list(df['response_pred'])
    data_dict['output_gt']=list(df['response_gt'])
    if "corpus_key" not in df.columns:
        df = pd.read_json("../annotations/instruction/scan2cap/scanrefer_val.json")
        df['corpus_key'] = df['scan_id'].astype(str) + "|" + df['target_id'].astype(str) + "|" + df['instance_type'].str.replace(" ", "_")

        
    data_dict['corpus_key']=list(df['corpus_key'])

    return data_dict

def evaluate_predictions_scanqa(json_data, ):
    """Evaluate predictions using the ScanQAEvaluator."""
    # Initialize the evaluator
    evaluator = ScanQAEvaluator()

    
    # Prepare the data
    data_dict = prepare_data_for_evaluation(json_data)
    
    # Update the evaluator with the data
    evaluator.update(data_dict)
    
    # Record the results (This can be done after all updates are complete)
    is_best, eval_results = evaluator.record(split='test', is_main_process=True)
    
    # Return the evaluation results
    return eval_results

def evaluate_predictions_sqa3d(json_data, ):
    """Evaluate predictions using the ScanQAEvaluator."""
    # Initialize the evaluator
    evaluator = SQA3DEvaluator()
    
    # Prepare the data
    data_dict = prepare_data_for_evaluation_sqa3d(json_data)
    
    # Update the evaluator with the data
    evaluator.update(data_dict)
    
    # Record the results (This can be done after all updates are complete)
    is_best, eval_results = evaluator.record(split='test', is_main_process=True)
    
    # Return the evaluation results
    return eval_results

def evaluate_predictions_cap(json_data, task):
    """Evaluate predictions using the ScanQAEvaluator."""
    # Initialize the evaluator
    evaluator = CaptionEvaluator(task)
    
    # Prepare the data
    if task =="scan2cap":
        data_dict = prepare_data_for_evaluation_scan2cap(json_data)
    else:
        data_dict = prepare_data_for_evaluation(json_data)
    
    # Update the evaluator with the data
    evaluator.update(data_dict)
    
    # Record the results (This can be done after all updates are complete)
    is_best, eval_results = evaluator.record(split='test', is_main_process=True)
    
    # Return the evaluation results
    return eval_results


# Mapping of number words to digits
word_to_num = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9
}

def extract_number(text):
    # Lowercase for matching
    text = text.lower()

    # Check for digit first (e.g. "6", "3")
    digit_match = re.search(r"\b([0-9])\b", text)
    if digit_match:
        return str(int(digit_match.group(1)))

    # Check for word (e.g. "five")
    for word, num in word_to_num.items():
        if re.search(rf"\b{word}\b", text):
            return str(num)

    # If no match found
    return text



# Example usage
if __name__ == "__main__":
    # Path to your JSON file
    #df_mine = pd.read_json('v8/results_scanQA_leo_mask.json')
    # df_mine = pd.read_json('v8/results_scanQA_v8_pretrained.json')
    tasks_ll3da = ["3dllm-dialogue",
              "3dllm-embodied-planning",
             "3dllm-scene-description",
              "nr3d_densecap",
              "scanQA",
             "scan2cap"
             ]
    tasks_v8_ll3da = ["dialog_3dllm",
              "plan_3dllm",
             "scene_cap_3dllm",
              "nr3d",
              "scanqa",
             "scan2cap",
            "scene_cap_3dllm"
             ]

    tasks_leo = [
              "sqa3d",
              "scanqa",
             "scan2cap"
             ]
    ll3da_paths = [f"ll3da/results_{i}_ll3da.json" for i in tasks_ll3da]
    v8_paths = [ "v8/results_scanQA_v8_pretrained.json", f"v8/results_scan2cap_v8_pretrained.json"]
    v8_ROI15_paths = [ "../training_results/v8_pretrained/GS_v8_tune_ROI15_real15/eval_results/gs_sqa3d/results.json", 
                      "../training_results/v8_pretrained/GS_v8_tune_ROI15_real15/eval_results/gs_scan2cap/results.json",
                     "../training_results/v8_pretrained/GS_v8_tune_ROI15_real15/eval_results/gs_scanqa/results.json"]
    


    
    ablations_depth_wise = [ f"ablation_depth_wise/gs_{task}/results.json" for task in tasks_v8_ll3da]
    ablations_knn = [ f"ablation_knn/gs_{task}/results.json" for task in tasks_v8_ll3da]
    ablations_prompt_llm = [ f"../training_results/ablations_prompt_to_llm_ll3da/GS_gs_v8_ll3da_ablate_prompt_llm_train/eval_results/gs_{task}/results.json" for task in tasks_v8_ll3da]
    ablation_no_task_spars = [ f"ablation_no_task-spars/gs_{task}/results.json" for task in tasks_v8_ll3da]
    ablation_learnable_queries = [ f"../training_results/ablations_learnable_queries_ll3da/GS_gs_v8_ll3da_ablate_queries_train/eval_results/gs_{task}/results.json" for task in tasks_v8_ll3da]
    ablation_downsampling = [ f"../training_results/only_for_eval/eval_results_ablate_downsampling_ll3da/gs_{task}/results.json" for task in tasks_v8_ll3da]
    ablation_no_roi = [ f"ablation_roi/gs_{task}/results.json" for task in tasks_v8_ll3da]
    ablation_no_spars = [ f"../training_results/ablate_no_spars/GS_gs_v8_ll3da_ablate_no_spars/eval_results/gs_{task}/results.json" for task in tasks_v8_ll3da]

    
    
    leo_mask_iou = [f"../training_results/LEO-1_leo_mask/eval_results/leo_mask_{i}_iou.json" for i in tasks_leo]
    ll3da_paths_no_iou = [f"ll3da_no_iou/results_{i}_ll3da.json" for i in tasks_ll3da]
    gs_v8_ll3da_paths = [f"../training_results/gs_v8_ll3da/GS_gs_v8_ll3da_align/eval_results/gs_{p}/results.json" for p in tasks_v8_ll3da]
    ll3da_planning_no_iou = ["results_3dllm-embodied-planning_ll3da.json", "results_scan2cap_ll3da.json"]

    '''
    #for p, task in zip(ll3da_paths_no_iou, tasks_ll3da):
    for p, task in zip(ablation_no_roi, tasks_v8_ll3da):
    #for p, task in zip(v8_ROI15_paths, ["sqa3d", "scan2cap", "scanqa"]):
    #for p, task in zip(leo_mask_iou, tasks_leo):
    #for p, task in zip(ll3da_planning_no_iou, ["plan_3dllm", "scan2cap"]):
    #for p, task in zip(["obj_presence.json"],["scanqa"]):
        df_mine = pd.read_json(p)
        print(f"_______Task: {task}_____")
        # Evaluate the predictions
        if task in ["scanQA","scanqa"]:
            eval_results = evaluate_predictions_scanqa(df_mine)
        elif task in ["sqa3d"]:
            eval_results = evaluate_predictions_sqa3d(df_mine)
        else:
            eval_results = evaluate_predictions_cap(df_mine, task)
        
        # Print the evaluation results
        print(json.dumps(eval_results, indent=2))
    '''

    task ="scanqa"
    models = {"ll3da": f"../../LL3DA/ckpts-v0_backup/opt-1.3b/ll3da-generalist/qa_leo_structure.json",
              "ours": f"../training_results/ablations_downsampling_ll3da/GS_gs_v8_ll3da_ablate_downsampling_train/eval_results/gs_scanqa/results.json"}
    for m,p in models.items():
        df_mine = pd.read_json(p)
        str_check = "where"
        str_check_not = "aaafafsadf"
        df_mine=df_mine[df_mine["instruction"].str.contains(str_check, na=False) & ~df_mine["instruction"].str.contains(str_check_not, na=False)]


        print(f"_______Task: {m}, {str_check}_____")
        # Evaluate the predictions
        if task in ["scanQA","scanqa"]:
            eval_results = evaluate_predictions_scanqa(df_mine)
        elif task in ["sqa3d"]:
            eval_results = evaluate_predictions_sqa3d(df_mine)
        else:
            eval_results = evaluate_predictions_cap(df_mine, task)
        
        # Print the evaluation results
        print(json.dumps(eval_results, indent=2))
    print(r)
    #for p, task in zip(ll3da_paths_no_iou, tasks_ll3da):
    '''
    a=None
    b=None
    for m,p in models.items():
    #for p, task in zip(v8_ROI15_paths, ["sqa3d", "scan2cap", "scanqa"]):
    #for p, task in zip(leo_mask_iou, tasks_leo):
    #for p, task in zip(ll3da_planning_no_iou, ["plan_3dllm", "scan2cap"]):
    #for p, task in zip(["obj_presence.json"],["scanqa"]):
        df_mine = pd.read_json(p)
        print(f"_______Task: {m}_____")
        df_mine["response_pred2"] = df_mine["response_pred"].apply(extract_number)
        df_mine["response_gt2"] =df_mine["response_gt"].apply(lambda x: str(x[0]))
        print(len(df_mine[df_mine["response_gt"]==df_mine["response_pred2"]]))
        if a is None:
            df_mine["question"] =df_mine["question"].apply(lambda x: x.split(": ")[1].split(" #")[0])
            df_mine.set_index(["question", "scene"])
            
            # Boolean Series: matches in df1
            a = df_mine["response_pred2"] == df_mine["response_gt2"]
            b=df_mine

            value_counts = df_mine.loc[a, "response_gt2"].value_counts()

            print("Value counts of response_gt where ll3da prediction matched:")
            print(value_counts)
        else:
            print(df_mine.columns)
            df_mine["question"] =df_mine["instruction"].apply(lambda x: x.split(": ")[-1])
            df_mine["scene"] =df_mine["scene_id"]
            df2_indexed = df_mine.set_index(["question", "scene"])
            

            # Step 2: Reorder df1 to match df2's index
            df1_aligned = b.loc[df2_indexed.index].reset_index()
            
            # Step 3: Reset df2 index as well (to keep everything clean and aligned)
            df2_aligned = df2_indexed.reset_index()
            # Boolean Series: matches in df2
            match_df2 = df2_aligned["response_pred2"] == df2_aligned["response_gt2"]
            value_counts = df2_aligned.loc[match_df2, "response_gt2"].value_counts()

            print("Value counts of response_gt where our prediction matched:")
            print(value_counts)

            import pandas as pd

            # Step 2: Get indices of matched rows
            matched_indices = df2_aligned[match_df2].index
            
            # Step 3: Get indices of non-matched rows
            non_matched_indices = df2_aligned[~match_df2].index
            
            # Step 4: Randomly sample 800 non-matching indices
            random_sample_indices = non_matched_indices.to_series().sample(n=800, random_state=42).index
            
            # Step 5: Combine matched + random indices
            final_indices = matched_indices.union(random_sample_indices)
            
            # Step 6: Create a full boolean mask
            final_mask = df_mine.index.isin(final_indices)
            
            # Apply the mask to both DataFrames
            df_combined = df2_aligned[final_mask]
            ll3da_combined = df1_aligned[final_mask]

            # Step 5: Save to JSON
            df_combined.to_json("counts_dataset_v2_ours.json", orient="records")
            ll3da_combined.to_json("counts_dataset_v2_ll3da.json", orient="records")
            
            print(f"Saved {len(df_combined)} rows to counts_dataset_v2.json")

            
            # Rows where df1 matches but df2 does not
            condition = match_df2 & (~a)

            # Count of such rows
            print(f"Ours correct but ll3da not: {condition.sum()}")
        '''
    import pandas as pd
    import numpy as np
    
    # Initialize for first model
    a = None
    b = None
    
    # Loop through model name and path
    for model_name, path in models.items():
        print(f"\n_______ Task: {model_name} _______")
    
        # Load JSON
        df_mine = pd.read_json(path)
    
        # Preprocess predictions
        df_mine["response_pred2"] = df_mine["response_pred"].apply(extract_number)
        df_mine["response_gt2"] = df_mine["response_gt"].apply(lambda x: str(x[0]))
    
        print(f"Correct predictions: {(df_mine['response_gt2'] == df_mine['response_pred2']).sum()}")
    
        if a is None:
            # Prepare and index the first reference DataFrame (ll3da)
            df_mine["question"] = df_mine["question"].apply(lambda x: x.split(": ")[1].split(" #")[0])
            df_mine.set_index(["question", "scene"], inplace=True)
    
            # Save the match condition and reference DataFrame
            a = df_mine["response_pred2"] == df_mine["response_gt2"]
            b = df_mine.copy()
    
            # Print value counts of correct responses
            print("Value counts of response_gt where ll3da prediction matched:")
            print(df_mine.loc[a, "response_gt2"].value_counts())
    
        else:
            # Prepare question and scene keys for current model
            df_mine["question"] = df_mine["instruction"].apply(lambda x: x.split(": ")[-1])
            df_mine["scene"] = df_mine["scene_id"]
            df2_indexed = df_mine.set_index(["question", "scene"])
    
            # Align previous df1 (ll3da) to current df2
            b_indexed = b.copy()
            df1_aligned = b_indexed.loc[df2_indexed.index].reset_index()
            df2_aligned = df2_indexed.reset_index()
    
            # Compute match condition for current model
            match_df2 = df2_aligned["response_pred2"] == df2_aligned["response_gt2"]
            match_df1 = df1_aligned["response_pred2"] == df1_aligned["response_gt2"]
            print("Value counts of response_gt where our prediction matched:")
            print(df2_aligned.loc[match_df2, "response_gt2"].value_counts())
    
            # Get matched and unmatched indices
            matched_indices = df2_aligned[match_df2].index
            non_matched_indices = df2_aligned[~match_df2&~match_df1].index
    
            # Randomly sample 800 non-matching rows
            # random_sample_indices = non_matched_indices.to_series().sample(n=800, random_state=42).index
            # final_indices = matched_indices.union(random_sample_indices)
            # Get non-matched df2 rows that are also not matched by df1
            non_matched_df2 = df2_aligned.loc[~match_df2 & ~match_df1]
            
            # Split based on response_gt
            non_gt_not_2 = non_matched_df2[non_matched_df2["response_gt2"] != "2"]
            non_gt_2 = non_matched_df2[non_matched_df2["response_gt2"] == "2"]
            
            # Sample up to 800 from the "not 2" category
            sample_needed = 800
            sample_from_not_2 = non_gt_not_2.sample(
                min(len(non_gt_not_2), sample_needed),
                random_state=42
            )
            
            # If needed, sample remaining from "2"
            remaining = sample_needed - len(sample_from_not_2)
            if remaining > 0:
                sample_from_2 = non_gt_2.sample(min(len(non_gt_2), remaining), random_state=42)
            else:
                sample_from_2 = pd.DataFrame(columns=non_matched_df2.columns)
            
            # Combine both samples
            random_sample_df = pd.concat([sample_from_not_2, sample_from_2])
            random_sample_indices = random_sample_df.index
            
            # Final set of indices = matched + sampled
            final_indices = matched_indices.union(random_sample_indices)
    
            # Mask for selected rows
            final_mask = df2_aligned.index.isin(final_indices)
    
            # Apply mask to both aligned DataFrames
            df_combined = df2_aligned[final_mask]
            ll3da_combined = df1_aligned[final_mask]
    
            # Save to JSON
            df_combined.to_json("counts_dataset_v4_ours.json", orient="records")
            ll3da_combined.to_json("counts_dataset_v4_ll3da.json", orient="records")
            print(f"Saved {len(df_combined)} rows to counts_dataset_v2_ours.json and counts_dataset_v2_ll3da.json")
    
            # Compare prediction correctness
            ll3da_match_mask = a.loc[df2_indexed.index].reset_index(drop=True)
            condition = match_df2 & (~ll3da_match_mask)
            print(f"Ours correct but ll3da not: {condition.sum()}")

            df_mine["response_gt2"] = df_mine["response_gt"].apply(lambda x: str(x[0]))
    
            print(f"Correct predictions on NEW, LL3DA: {(ll3da_combined['response_gt2'] == ll3da_combined['response_pred2']).sum()}")
            print(f"Correct predictions on NEW, Ours: {(df_combined['response_gt2'] == df_combined['response_pred2']).sum()}")

    
        # # Evaluate the predictions
        # if task in ["scanQA","scanqa"]:
        #     eval_results = evaluate_predictions_scanqa(df_mine)
        # elif task in ["sqa3d"]:
        #     eval_results = evaluate_predictions_sqa3d(df_mine)
        # else:
        #     eval_results = evaluate_predictions_cap(df_mine, task)
        
        # # Print the evaluation results
        # print(json.dumps(eval_results, indent=2))