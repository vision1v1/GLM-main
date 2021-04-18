# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Race."""

from tasks.eval_utils import accuracy_func_provider
from finetune_gpt2 import finetune
from tasks.superglue.dataset import GlueDataset, CLASSIFICATION_DATASETS, MULTI_CHOICE_DATASETS, PROCESSORS, \
    get_output_func
from tasks.superglue.evaluate import qa_exact_match, qa_f1, multirc_em
from collections import OrderedDict
from tasks.eval_utils import accuracy_metric, f1_macro_metric, f1_metric
from tasks.superglue.pvp import PVPS

default_metrics = {
    "record": [("EM", qa_exact_match), ("F1", qa_f1)],
    "copa": [("accuracy", accuracy_metric)],
    "rte": [("accuracy", accuracy_metric)],
    "boolq": [("accuracy", accuracy_metric)],
    "wic": [("accuracy", accuracy_metric)],
    "wsc": [("accuracy", accuracy_metric)],
    "wsc1": [("accuracy", accuracy_metric)],
    "cb": [("accuracy", accuracy_metric), ("f1-macro", f1_macro_metric)],
    "multirc": [("f1a", f1_metric), ("em", multirc_em), ("acc", accuracy_metric)]
}


def train_valid_datasets_provider(args, tokenizer):
    """Provide train and validation datasets."""
    train_dataset = GlueDataset(args, "train", tokenizer)
    valid_dataset = GlueDataset(args, "dev", tokenizer, for_train=True)

    return train_dataset, valid_dataset


def metrics_func_provider(args, tokenizer, is_test):
    """Privde metrics callback function."""

    def single_dataset_provider(split):
        return GlueDataset(args, split, tokenizer)

    output_func = get_output_func(args.task.lower(), args)
    eval_func = None
    if args.task.lower() == 'wsc' and args.cloze_eval and not args.wsc_negative:
        from tasks.language_model.finetune import classify_evaluate
        eval_func = classify_evaluate
    metric_dict = OrderedDict(default_metrics[args.task.lower()])
    return accuracy_func_provider(single_dataset_provider, metric_dict, args, is_test=is_test, eval_func=eval_func,
                                  output_func=output_func, only_rank0=False)


def main(args):
    model_kwargs = {}
    processor = PROCESSORS[args.task.lower()](args)
    pvp = PVPS[args.task.lower()](args, None, processor.get_labels(), args.seq_length,
                                  pattern_id=args.pattern_id, is_multi_token=args.multi_token)
    if args.continuous_prompt:
        model_kwargs["spell_length"] = pvp.spell_length
    if args.task.lower() == 'wsc' and args.cloze_eval and not args.wsc_negative:
        from tasks.language_model.finetune import lm_forward_step
        finetune(args, train_valid_datasets_provider, model_kwargs,
                 end_of_epoch_callback_provider=metrics_func_provider, forward_step=lm_forward_step)
    else:
        if args.cloze_eval:
            multi_token = pvp.is_multi_token
        else:
            multi_token = args.task.lower() in MULTI_CHOICE_DATASETS
        args.multi_token = multi_token
        if not multi_token:
            model_kwargs["model_type"] = "multiple_choice" if args.cloze_eval else "classification"
            model_kwargs["multi_token"] = False
            model_kwargs["num_labels"] = len(processor.get_labels())
        else:
            model_kwargs["model_type"] = "multiple_choice"
            model_kwargs["multi_token"] = True
            model_kwargs["num_labels"] = 1
        finetune(args, train_valid_datasets_provider, model_kwargs,
                 end_of_epoch_callback_provider=metrics_func_provider)
