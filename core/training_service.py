import os.path
import confuse
import itertools
import sys
import logging
import os
import random
import warnings
import torch
import shutil
import json
import numpy as np
from confuse import Configuration
from math import ceil
from collections import OrderedDict
from transformers.models.bert import BertPreTrainedModel, BertModel, BertConfig
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_cli import to_runner
from abc import ABC, abstractmethod
from sklearn.metrics import f1_score, recall_score, precision_score
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from collections import Counter
from copy import deepcopy
from dataclasses_serialization.json import JSONSerializer
from sklearn.exceptions import UndefinedMetricWarning
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import Module, CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, TypeVar, Union, Set, ClassVar, Tuple, Type
from torch import Tensor
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
    AdamW,
    get_linear_schedule_with_warmup,
    get_constant_schedule,
)


def init_logger():
    FORMAT = "%(asctime)s %(levelname)s (%(threadName)s) %(module)s: %(message)s"
    logging.basicConfig(format=FORMAT, level=logging.INFO)


init_logger()
logger = logging.getLogger(__name__)

Dataset_T = TypeVar("Dataset_T", bound=Dataset)


@dataclass
class FieldDefinition:
    STORE_NAME_APPENDIX: ClassVar[str] = "_vocabulary"

    name: str
    is_sequence: bool
    is_multilabel: bool
    pad_token: Optional[str]
    default_value: Optional[str]
    cls_token: Optional[str]
    sep_token: Optional[str]
    bos_token: Optional[str]
    eos_token: Optional[str]
    special_tokens: List[str] = field(default_factory=list)
    vocab: Optional[Dict[str, int]] = None
    reverse_vocab: Optional[Dict[int, str]] = None

    def __post_init__(self):
        self.__refresh_special_tokens()
        if self.vocab:
            self.reverse_vocab = {x: y for y, x in self.vocab.items()}

    def __refresh_special_tokens(self):
        special_tokens = [
            self.pad_token,
            self.default_value,
            self.cls_token,
            self.sep_token,
            self.bos_token,
            self.eos_token,
        ]
        self.special_tokens: List[str] = [
            special_token
            for special_token in special_tokens
            if special_token is not None
        ]

    @classmethod
    def create_sequential_field(
            cls,
            name: str,
            multilabel: bool,
            tokenizer: PreTrainedTokenizer = None,
            default_label: str = None,
            vocabulary: Dict[str, int] = None,
    ):
        return cls.__create_field(
            name=name,
            is_sequence=True,
            is_multilabel=multilabel,
            tokenizer=tokenizer,
            default_label=default_label,
            vocab=vocabulary,
        )

    @classmethod
    def __create_field(
            cls,
            name: str,
            is_sequence: bool,
            is_multilabel: bool,
            tokenizer: PreTrainedTokenizer = None,
            default_label: str = None,
            vocab: Dict[str, int] = None,
    ):
        is_tokenizer_verbose = tokenizer.verbose if tokenizer else False
        if is_tokenizer_verbose:
            tokenizer.verbose = False
        default_label = default_label or (tokenizer.unk_token if tokenizer else None)
        field_definition = FieldDefinition(
            name=name,
            is_sequence=is_sequence,
            is_multilabel=is_multilabel,
            pad_token=tokenizer.pad_token
            if tokenizer and tokenizer.pad_token != "None"
            else None,
            default_value=default_label,
            cls_token=tokenizer.cls_token
            if tokenizer and tokenizer.cls_token != "None"
            else None,
            sep_token=tokenizer.sep_token
            if tokenizer and tokenizer.sep_token != "None"
            else None,
            bos_token=tokenizer.bos_token
            if tokenizer and tokenizer.bos_token != "None"
            else None,
            eos_token=tokenizer.eos_token
            if tokenizer and tokenizer.eos_token != "None"
            else None,
            vocab=vocab,
        )
        if is_tokenizer_verbose:
            tokenizer.verbose = True
        return field_definition

    def attn_mask_name(self):
        return "{}_attn_mask".format(self.name)

    def ctx_mask_name(self):
        return "{}_ctx_mask".format(self.name)

    def pad_idx(self):
        if self.vocab:
            return self.stoi(self.pad_token)
        else:
            raise Exception(
                f"Field {self.name} vocabulary not provided/created before requesting pad_idx"
            )

    def build_labels_vocab(
            self,
            instances: List[Dict[str, Union[str, List[str]]]] = None,
            direct_vocab: List[str] = None,
            min_support: Union[int, float] = 1,
            default_counts_for_min_support: bool = True,
    ):

        if self.is_sequence:
            self.__build_sequence_labelling_vocabulary(
                instances,
                direct_vocab,
                self.default_value,
                min_support,
                default_counts_for_min_support,
            )
        else:
            self.__build_classification_vocabulary(
                instances,
                direct_vocab,
                self.default_value,
                min_support,
                default_counts_for_min_support,
            )

    def __init_vocab_with_special_tokens(
            self, default_value: str, is_sequential_field: bool
    ):
        self.__update_unk_token(default_value)
        vocab: Dict[str, int] = dict()
        if is_sequential_field:
            if self.pad_token:
                vocab[self.pad_token] = len(vocab)
            else:
                logger.warning(
                    f"WARNING: no pad_token for field {self.name} (this might be ok if it is the desired behaviour)"
                )
            for special_token in self.special_tokens:
                if special_token not in vocab:
                    vocab[special_token] = len(vocab)
        else:
            if self.default_value:
                vocab[self.default_value] = len(vocab)
        return vocab

    def __add_vocab(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.reverse_vocab = {x: y for y, x in self.vocab.items()}

    def __build_sequence_labelling_vocabulary(
            self,
            instances: List[Dict[str, Union[str, List[str]]]],
            direct_vocab: List[str],
            default_value: str,
            min_support: Union[int, float],
            default_counts_for_min_support: bool = True,
    ):
        vocab = self.__init_vocab_with_special_tokens(
            default_value=default_value, is_sequential_field=True
        )

        if instances and direct_vocab:
            raise Exception(
                "Instances and direct_vocab cannot be provided at the same time, they are mutually exclusive"
            )
        elif instances:
            vocab = self.__fill_vocab_from_sequence_labelling_instances(
                vocab=vocab,
                instances=instances,
                min_support=min_support,
                default_counts_for_min_support=default_counts_for_min_support,
            )
        elif direct_vocab:
            logger.info(f"Using direct_vocab {direct_vocab} for field: {self.name}")
            for elem in direct_vocab:
                if elem not in vocab:
                    vocab[elem] = len(vocab)
        else:
            raise Exception(
                "Instances or direct_vocab must be provided to build a vocabulary"
            )
        self.__add_vocab(vocab=vocab)

    def __fill_vocab_from_sequence_labelling_instances(
            self,
            vocab: Dict[str, int],
            instances: List[Dict[str, Union[str, List[str]]]],
            min_support: Union[int, float],
            default_counts_for_min_support: bool = True,
    ):
        counter = Counter()
        for instance in instances:
            field_content: List[str] = instance[self.name]
            for value in field_content:
                if self.is_multilabel:
                    for csv_value in value.split(","):
                        counter[csv_value] += 1
                else:
                    counter[value] += 1
        values_with_min_support = self.__filter_by_min_support(
            label_counter=counter,
            min_support=min_support,
            default_counts_for_min_support=default_counts_for_min_support,
        )
        for value in values_with_min_support:
            if value not in vocab:
                vocab[value] = len(vocab)
        return vocab

    def __build_classification_vocabulary(
            self,
            instances: List[Dict[str, Union[str, List[str]]]],
            direct_vocab: List[str],
            default_value: str,
            min_support: Union[int, float],
            default_counts_for_min_support: bool,
    ):
        vocab = self.__init_vocab_with_special_tokens(
            default_value=default_value, is_sequential_field=False
        )
        if instances and direct_vocab:
            raise Exception(
                "Instances and direct_vocab cannot be provided at the same time, they are mutually exclusive"
            )
        elif instances:
            vocab = self.__fill_vocab_from_classification_instances(
                vocab=vocab,
                instances=instances,
                min_support=min_support,
                default_counts_for_min_support=default_counts_for_min_support,
            )
        elif direct_vocab:
            logger.info(f"Using direct_vocab {direct_vocab} for field: {self.name}")
            for elem in direct_vocab:
                if elem not in vocab:
                    vocab[elem] = len(vocab)
        else:
            raise Exception(
                "Instances or direct_vocab must be provided to build a vocabulary"
            )
        self.__add_vocab(vocab=vocab)

    def __fill_vocab_from_classification_instances(
            self,
            vocab: Dict[str, int],
            instances: List[Dict[str, List[str]]],
            min_support: Union[int, float],
            default_counts_for_min_support: bool,
    ):
        counter = Counter()
        for instance in instances:
            field_content: Union[str, List[str]] = instance[self.name]
            if isinstance(field_content, list):
                for value in field_content:
                    counter[value] += 1
            elif isinstance(field_content, str):
                counter[field_content] += 1

        values_with_min_support = self.__filter_by_min_support(
            label_counter=counter,
            min_support=min_support,
            default_counts_for_min_support=default_counts_for_min_support,
        )
        for value in values_with_min_support:
            if value not in vocab:
                vocab[value] = len(vocab)
        return vocab

    def __filter_by_min_support(
            self,
            label_counter: Counter,
            min_support: Union[int, float],
            default_counts_for_min_support: bool,
    ):
        if min_support is None:
            return list(label_counter.keys())
        elif isinstance(min_support, int):
            if min_support < 0:
                raise Exception(
                    f"Integer min_support must be greater or equal to zero, provided value: {min_support}"
                )
            values_with_min_support = [
                x for x, y in label_counter.most_common() if y >= min_support
            ]
            return values_with_min_support
        elif isinstance(min_support, float):
            if min_support < 0.0 or min_support > 1.0:
                raise Exception(
                    f"Float min_support must be in range [0-1], provided value: {min_support}"
                )
            total_labels = sum(label_counter.values())
            if (
                    self.default_value
                    and self.default_value in label_counter
                    and not default_counts_for_min_support
            ):
                total_labels = total_labels - label_counter[self.default_value]
            values_with_min_support = [
                x
                for x, y in label_counter.most_common()
                if y / total_labels >= min_support
            ]
            return values_with_min_support

    def __update_unk_token(self, new_default: str):
        self.default_value = new_default
        self.__refresh_special_tokens()

    def stoi(self, value: str):
        if value in self.vocab:
            return self.vocab[value]
        elif self.default_value is not None:
            return self.vocab[self.default_value]
        else:
            raise Exception(
                f"Non-existent vocabulary value:{value} requested for a vocabulary without default value."
            )

    def serialize_to_file(self, base_path: str, field_name: str):
        self_copy = deepcopy(self)
        self_copy.reverse_vocab = None
        serializable_dict = JSONSerializer.serialize(self_copy)
        path = os.path.join(base_path, f"{field_name}{self.STORE_NAME_APPENDIX}.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(serializable_dict))

    @classmethod
    def deserialize_from_file(cls, base_path, field_name: str):
        path = os.path.join(base_path, f"{field_name}{cls.STORE_NAME_APPENDIX}.json")
        with open(path, "r", encoding="utf-8") as f:
            deserialized_dict = json.loads(f.read())
        return JSONSerializer.deserialize(cls, deserialized_dict)


@dataclass
class ModelStepOutput:
    loss: Tensor
    prediction_scores: Optional[Dict[str, Tensor]] = field(default_factory=dict)
    gold_labels: Dict[str, Tensor] = field(default_factory=dict)

    __annotations__ = {
        "loss": Tensor,
        "prediction_scores": Optional[Dict[str, Tensor]],
        "gold_labels": Dict[str, Tensor],
    }


@dataclass
class DatasetsVocabsAndExtraResources:
    data_fields: List[FieldDefinition]
    train_dataset: Dataset_T
    dev_dataset: Dataset_T
    extra_resources: Optional[Dict[str, Any]] = None

    __annotations__ = {
        "data_fields": List[FieldDefinition],
        "train_dataset": Dataset_T,
        "dev_dataset": Dataset_T,
        "extra_resources": Optional[Dict[str, Any]],
    }


class BaseModelOutcomePostProcessor(ABC):
    @abstractmethod
    def process(
            self, gold_labels: Tensor, model_outcomes: Tensor
    ):
        pass


class BaseEvaluationMetric(ABC):
    def __init__(
            self,
            name: str,
            target_field: Optional[FieldDefinition],
            preprocessor: Optional[BaseModelOutcomePostProcessor],
    ):
        self.name = name
        self.target_field: FieldDefinition = target_field
        self.preprocessor = preprocessor

    def evaluate(
            self,
            gold_labels: Tensor = None,
            model_outcomes: Tensor = None,
            loss: Tensor = None,
    ):
        if self.preprocessor:
            gold_labels, model_outcomes = self.preprocessor.process(
                gold_labels=gold_labels, model_outcomes=model_outcomes
            )
        return self.calculate_metric_value(
            gold_labels=gold_labels, model_outcomes=model_outcomes, loss=loss
        )

    @abstractmethod
    def calculate_metric_value(
            self,
            gold_labels: Tensor = None,
            model_outcomes: Tensor = None,
            loss: Tensor = None,
    ):
        pass


class EvaluationMetricCombination:
    def __init__(
            self,
            name: str,
            metrics_to_combine: Tuple[BaseEvaluationMetric, ...],
            weights: Optional[Tuple[float, ...]] = None,
    ):
        if weights and len(metrics_to_combine) != len(weights):
            raise Exception(
                f"Weights for metric combination ({name}) have been defined but do not match in number: "
                f"{len(weights)} for {len(metrics_to_combine)} metrics"
            )

        self.name = name
        self.metrics_to_combine = metrics_to_combine
        self.weights = weights

    def combine(self, metric_names_and_values: Dict[str, float]):
        values_to_combine = [
            metric_names_and_values[metric.name] for metric in self.metrics_to_combine
        ]
        if self.weights:
            values_to_combine = [
                value * self.weights[i] for i, value in enumerate(values_to_combine)
            ]
        averaged_values = sum(values_to_combine) / len(values_to_combine)
        return averaged_values


class GenericLossMetric(BaseEvaluationMetric):
    def calculate_metric_value(
            self,
            gold_labels: Tensor = None,
            model_outcomes: Tensor = None,
            loss: Tensor = None,
    ):
        return loss


class BaseEvaluationMetricsAccumulator(ABC):
    @abstractmethod
    def calculate_and_accumulate_metrics(
            self, gold_labels, model_outcomes, loss
    ):
        pass

    @abstractmethod
    def reset_metric_accumulation(self, metric_name):
        pass

    @abstractmethod
    def get_averaged_metric_value(self, metric_name):
        pass

    @abstractmethod
    def get_all_metrics_average(self, reset_afterwards=False):
        pass

    @abstractmethod
    def reset_all_metrics_accumulation(self):
        pass


class SimpleEvaluationMetricsAccumulator(BaseEvaluationMetricsAccumulator):
    logger: ClassVar[logging.Logger] = logging.getLogger(__name__)

    def __init__(
            self,
            evaluation_metrics: List[
                Union[BaseEvaluationMetric, EvaluationMetricCombination]
            ],
    ):
        self.evaluation_metrics: List[BaseEvaluationMetric] = [
            ev for ev in evaluation_metrics if isinstance(ev, BaseEvaluationMetric)
        ]
        self.metric_combinations: List[EvaluationMetricCombination] = [
            ev
            for ev in evaluation_metrics
            if isinstance(ev, EvaluationMetricCombination)
        ]
        self.__check_and_add_combination_metrics()
        self.accumulated_metric_values = {
            metric.name: 0 for metric in evaluation_metrics
        }
        self.accumulated_metric_counts = {
            metric.name: 0 for metric in evaluation_metrics
        }
        self.logger.info(
            f"Evaluation metrics configured in the metrics accumulator:{[m.name for m in self.evaluation_metrics]}"
        )

    def __check_and_add_combination_metrics(self):
        for metric_combination in self.metric_combinations:
            for metric in metric_combination.metrics_to_combine:
                if metric not in self.evaluation_metrics:
                    self.evaluation_metrics.append(metric)

    def __accumulate_metric(self, metric_name, metric_value):
        if metric_name in self.accumulated_metric_values:
            self.accumulated_metric_values[metric_name] += metric_value
            self.accumulated_metric_counts[metric_name] += 1
        else:
            self.accumulated_metric_values[metric_name] = metric_value
            self.accumulated_metric_counts[metric_name] = 1

    def calculate_and_accumulate_metrics(
            self, gold_labels, model_outcomes, loss
    ):
        calculated_metric_values = []
        for evaluation_metric in self.evaluation_metrics:
            if evaluation_metric.target_field:
                target_logits = model_outcomes[evaluation_metric.target_field.name]
                target_labels = gold_labels[evaluation_metric.target_field.name]
                metric_value = evaluation_metric.evaluate(
                    target_labels, target_logits, loss
                )
            else:
                metric_value = evaluation_metric.evaluate(
                    gold_labels, model_outcomes, loss
                )
            self.__accumulate_metric(evaluation_metric.name, metric_value)
            calculated_metric_values.append((evaluation_metric.name, metric_value))
        calculated_metric_values_as_dict = {
            name: value for name, value in calculated_metric_values
        }
        for metric_combination in self.metric_combinations:
            value = metric_combination.combine(
                {
                    m.name: calculated_metric_values_as_dict[m.name]
                    for m in metric_combination.metrics_to_combine
                }
            )
            self.__accumulate_metric(metric_combination.name, value)
            calculated_metric_values.append((metric_combination.name, value))
        return calculated_metric_values

    def reset_metric_accumulation(self, metric_name):
        self.accumulated_metric_values[metric_name] = 0
        self.accumulated_metric_counts[metric_name] = 0

    def get_averaged_metric_value(self, metric_name):
        if (
                metric_name in self.accumulated_metric_counts
                and self.accumulated_metric_counts[metric_name] > 0
        ):
            return (
                    self.accumulated_metric_values[metric_name]
                    / self.accumulated_metric_counts[metric_name]
            )
        else:
            return 0.0

    def get_all_metrics_average(self, reset_afterwards=False):
        all_metrics_average = {}
        for metric_name in self.accumulated_metric_values.keys():
            all_metrics_average[metric_name] = self.get_averaged_metric_value(
                metric_name
            )
        if reset_afterwards:
            self.reset_all_metrics_accumulation()
        return all_metrics_average

    def reset_all_metrics_accumulation(self):
        for metric_name in self.accumulated_metric_values.keys():
            self.reset_metric_accumulation(metric_name)


@dataclass
class EvaluationMetricsHolder:
    logger: ClassVar[logging.Logger] = logging.getLogger(__name__)

    evaluation_metrics: List[Union[BaseEvaluationMetric, EvaluationMetricCombination]]
    metric_to_check: Union[BaseEvaluationMetric, EvaluationMetricCombination]
    maximize_checked_metric: bool
    metrics_in_progress_bar: List[
        Union[BaseEvaluationMetric, EvaluationMetricCombination]
    ]
    metrics_in_checkpoint_name: List[
        Union[BaseEvaluationMetric, EvaluationMetricCombination]
    ]

    __annotations__ = {
        "evaluation_metrics": List[
            Union[BaseEvaluationMetric, EvaluationMetricCombination]
        ],
        "metric_to_check": Union[BaseEvaluationMetric, EvaluationMetricCombination],
        "maximize_checked_metric": bool,
        "metrics_in_progress_bar": List[
            Union[BaseEvaluationMetric, EvaluationMetricCombination]
        ],
        "metrics_in_checkpoint_name": List[
            Union[BaseEvaluationMetric, EvaluationMetricCombination]
        ],
    }

    def __post_init__(self):
        self.__add_default_loss()
        self.logger.info(
            f"Evaluation metrics: {[metric.name for metric in self.evaluation_metrics]}"
        )
        self.logger.info(
            f'Evaluation metric to {"MAXIMIZE" if self.maximize_checked_metric else "MINIMIZE"} '
            f"for checkpoint-saving/early-stopping: {self.metric_to_check.name}"
        )

    def __add_default_loss(self):
        loss_metric = GenericLossMetric(
            name="loss", target_field=None, preprocessor=None
        )
        self.evaluation_metrics.append(loss_metric)
        self.metrics_in_progress_bar.append(loss_metric)
        self.metrics_in_checkpoint_name.append(loss_metric)


class PrecisionMetric(BaseEvaluationMetric):
    def __init__(
            self,
            name: str,
            target_field: Optional[FieldDefinition],
            preprocessor: Optional[BaseModelOutcomePostProcessor],
            average_method: str,
            labels_to_evaluate: Optional[List[int]] = None,
            binarization_negative_labels: Optional[List[int]] = None,
    ):
        super().__init__(name, target_field, preprocessor)
        self.labels_to_evaluate = labels_to_evaluate
        self.binarization_negative_labels = binarization_negative_labels
        if self.labels_to_evaluate and self.binarization_negative_labels:
            raise Exception(
                f"labels_to_evaluate and binarization_negative_labels cannot be set at the same time"
            )
        self.average_method = average_method

    def calculate_metric_value(
            self,
            gold_labels: Tensor = None,
            model_outcomes: Tensor = None,
            loss: Tensor = None,
    ):
        if self.binarization_negative_labels:
            bin_gold_labels, bin_model_outcomes = binarize_evaluation_values(
                gold_labels=gold_labels,
                model_outcomes=model_outcomes,
                negative_binarization_labels=self.binarization_negative_labels,
            )
            return precision_score(
                numpy_friendly(bin_gold_labels),
                numpy_friendly(bin_model_outcomes),
                pos_label=1,
                average="binary",
            )
        else:
            return precision_score(
                numpy_friendly(gold_labels),
                numpy_friendly(model_outcomes),
                labels=self.labels_to_evaluate,
                average=self.average_method,
            )


class RecallMetric(BaseEvaluationMetric):
    def __init__(
            self,
            name: str,
            target_field: Optional[FieldDefinition],
            preprocessor: Optional[BaseModelOutcomePostProcessor],
            average_method: str,
            labels_to_evaluate: Optional[List[int]] = None,
            binarization_negative_labels: Optional[List[int]] = None,
    ):
        super().__init__(name, target_field, preprocessor)
        self.labels_to_evaluate = labels_to_evaluate
        self.binarization_negative_labels = binarization_negative_labels
        if self.labels_to_evaluate and self.binarization_negative_labels:
            raise Exception(
                f"labels_to_evaluate and binarization_negative_labels cannot be set at the same time"
            )
        self.average_method = average_method

    def calculate_metric_value(
            self,
            gold_labels: Tensor = None,
            model_outcomes: Tensor = None,
            loss: Tensor = None,
    ):
        if self.binarization_negative_labels:
            bin_gold_labels, bin_model_outcomes = binarize_evaluation_values(
                gold_labels=gold_labels,
                model_outcomes=model_outcomes,
                negative_binarization_labels=self.binarization_negative_labels,
            )
            return recall_score(
                numpy_friendly(bin_gold_labels),
                numpy_friendly(bin_model_outcomes),
                pos_label=1,
                average="binary",
            )
        else:
            return recall_score(
                numpy_friendly(gold_labels),
                numpy_friendly(model_outcomes),
                labels=self.labels_to_evaluate,
                average=self.average_method,
            )


class FscoreMetric(BaseEvaluationMetric):
    def __init__(
            self,
            name: str,
            target_field: Optional[FieldDefinition],
            preprocessor: Optional[BaseModelOutcomePostProcessor],
            average_method: str,
            labels_to_evaluate: Optional[List[int]] = None,
            binarization_negative_labels: Optional[List[int]] = None,
            multilabel_threshold: Optional[float] = None,
    ):
        super().__init__(name, target_field, preprocessor)
        self.labels_to_evaluate = labels_to_evaluate
        self.binarization_negative_labels = binarization_negative_labels
        if self.labels_to_evaluate and self.binarization_negative_labels:
            raise Exception(
                f"labels_to_evaluate and binarization_negative_labels cannot be set at the same time"
            )
        self.average_method = average_method
        self.multilabel_threshold = multilabel_threshold
        if self.binarization_negative_labels and self.multilabel_threshold:
            raise Exception(
                f"multilabel_threshold and binarization_negative_labels cannot be set at the same time"
            )

    def calculate_metric_value(
            self,
            gold_labels: Tensor = None,
            model_outcomes: Tensor = None,
            loss: Tensor = None,
    ):
        if self.binarization_negative_labels:
            bin_gold_labels, bin_model_outcomes = binarize_evaluation_values(
                gold_labels=gold_labels,
                model_outcomes=model_outcomes,
                negative_binarization_labels=self.binarization_negative_labels,
            )
            return f1_score(
                numpy_friendly(bin_gold_labels),
                numpy_friendly(bin_model_outcomes),
                pos_label=1,
                average="binary",
            )
        else:
            if self.multilabel_threshold:
                model_outcomes = model_outcomes.ge(self.multilabel_threshold).long()
            return f1_score(
                numpy_friendly(gold_labels),
                numpy_friendly(model_outcomes),
                labels=self.labels_to_evaluate,
                average=self.average_method,
            )


def numpy_friendly(t: Tensor):
    return t.detach().cpu()


def binarize_evaluation_values(
        gold_labels: Tensor, model_outcomes: Tensor, negative_binarization_labels: List[int]
):
    bin_gold_labels = torch.tensor(
        [label not in negative_binarization_labels for label in gold_labels.view(-1)]
    ).view(gold_labels.shape)
    bin_model_outcomes = torch.tensor(
        [label not in negative_binarization_labels for label in model_outcomes.view(-1)]
    ).view(model_outcomes.shape)
    return bin_gold_labels, bin_model_outcomes


EXTRA_RESOURCE_APPENDIX = "_extra_resource"


class MapaEntityDetectionTrainerConfig(BaseModel):
    model_name: str = Field(..., description="The name of the trained model")
    model_version: int = Field(..., description="The version number of the model")
    model_description: Optional[str] = Field(
        description="An optional description of the model"
    )

    pretrained_model_name_or_path: Union[str] = Field(
        ..., description="The name or path to the pretrained base model"
    )
    pretrained_tokenizer_name_or_path: Optional[str] = Field(
        default=None,
        description="Name/path to pretrained tokenizer, "
                    "only in case it differs from the model name/path",
    )
    do_lower_case: bool = Field(
        default=False, description="Force the tokenizer's do_lower_case to True"
    )

    train_set: str = Field(..., description="Name of the training data file")
    dev_set: str = Field(
        ..., description="Name of the validation (a.k.a dev) data file"
    )

    checkpoints_folder: Union[Path, str] = Field(
        ..., description="Directory to store the model checkpoints during the training"
    )
    max_checkpoints_to_keep: Optional[int] = Field(
        None,
        description="Number of checkpoints to keep during training (the oldest ones are removed)",
    )

    num_epochs: int = Field(200, description="The maximum number of epochs to train")
    early_stopping_patience: int = Field(
        50,
        description="Number of epochs without improvement before early-stopping the training",
    )

    batch_size: int = Field(
        ...,
        description="The batch size (number of examples processed at once) during the training",
    )

    train_iters_to_eval: int = Field(
        default=-1,
        description="The number of iterations before triggering an evaluation (apart from full epochs)",
    )

    gradients_accumulation_steps: int = Field(
        default=1,
        description="The number of steps to accumulate gradients before an optimizer step",
    )
    clip_grad_norm: float = Field(
        default=1.0, description="The value for gradient clipping"
    )

    lr: float = Field(2e-05, description="The learning rate for the training")
    warmup_epochs: float = Field(
        default=1.0,
        description="The number of epochs to warmup the learning rate. Admits non-integer values, e.g. 0.5",
    )

    amp: bool = Field(
        default=True,
        description="The use of Automatic Mixed Precision (amp) to leverage fp16 operations and speed-up GPU training",
    )

    random_seed: int = Field(
        default=42, description="The random seed, to make the experiments reproducible"
    )

    valid_seq_len: int = Field(
        default=300,
        description="The length of the sliding-window (the central valid part excluding the contexts)",
    )
    ctx_len: int = Field(
        default=100,
        description="The length of the context surrounding each sliding-window",
    )
    labels_to_omit: str = Field(
        default="O,X,[CLS],[SEP],[PAD]",
        description="The labels omitted from the in-training evaluation (default for BERT)",
    )

    @classmethod
    def parse_config_from_console(
            cls, exit_on_error: bool = True
    ):
        class ConfigCatcher:
            def __init__(self):
                self.parsed_config: Optional[MapaEntityDetectionTrainerConfig] = None

            def catch_config(self):
                def launch(config: MapaEntityDetectionTrainerConfig):
                    self.parsed_config = config
                    return 0

                def handle_exceptions(exception: BaseException):
                    logger.error(f"{exception}")
                    return 0

                to_runner(cls, launch, exception_handler=handle_exceptions)(
                    sys.argv[1:]
                )
                return self.parsed_config

        parsed_config = ConfigCatcher().catch_config()
        if not parsed_config and exit_on_error:
            sys.exit(1)
        return parsed_config

    def pretty_print_config(self):
        params = []
        for arg in vars(self):
            params.append(f"{arg}: {getattr(self, arg)}")
        return (
                "\n"
                "==================================================\n"
                "  Experiment configuration and hyper-parameters:"
                "\n"
                "==================================================\n  "
                + "\n  ".join(params)
                + "\n================================================"
        )


class ModelSaver:
    logger: ClassVar[logging.Logger] = logging.getLogger(__name__)

    def __init__(
            self,
            base_folder,
            model_name: str,
            model_version: int,
            data_fields: Dict[str, FieldDefinition],
            extra_resources: Dict[str, Any],
            training_config: MapaEntityDetectionTrainerConfig,
            tokenizer: PreTrainedTokenizer = None,
    ):
        self.base_folder = base_folder
        self.model_name = model_name
        self.model_version = model_version
        self.data_fields: Dict[str, FieldDefinition] = data_fields
        self.extra_resources = extra_resources
        self.training_config_as_dict = {
            x: str(y) if isinstance(y, Path) else y
            for x, y in vars(training_config).items()
        }
        self.tokenizer: PreTrainedTokenizer = tokenizer

    def save_checkpoint(
            self,
            model: PreTrainedModel,
            epoch: int,
            iteration: int,
            metrics: Dict[str, float],
    ):
        model_folder_name = self.__compose_model_folder_name(
            epoch=epoch, iteration=iteration, metrics=metrics
        )
        model_folder = self.__check_and_obtain_model_folder_path(model_folder_name)
        self.logger.info(f"Saving model to {os.path.abspath(model_folder)}")
        model.save_pretrained(save_directory=model_folder)
        if self.tokenizer:
            self.logger.info("Saving tokenizer together with the model")
            self.tokenizer.save_pretrained(model_folder)
        if self.data_fields:
            self.__store_data_fields(model_folder)
        if self.extra_resources:
            self.__store_extra_resources(model_folder)
        self.__store_configuration_info(model_folder)
        return os.path.abspath(model_folder)

    def __check_and_obtain_model_folder_path(self, model_folder_name):
        model_folder = os.path.join(self.base_folder, model_folder_name)
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        return model_folder

    def __compose_model_folder_name(self, epoch, iteration, metrics: Dict[str, float]):
        composed_name = (
            f"{self.model_name}_v{self.model_version}_epoch{epoch}_iter{iteration}"
        )
        for x, y in metrics.items():
            composed_name += f"_{x}-{y:1.4f}"
        return composed_name

    def __store_data_fields(self, model_folder):
        self.logger.info(f"Saving vocabularies: {self.data_fields.keys()}")
        for vocab_name, data_field in self.data_fields.items():
            data_field.serialize_to_file(base_path=model_folder, field_name=vocab_name)

    def __store_extra_resources(self, model_folder):
        self.logger.info(f"Saving extra_resources: {self.extra_resources.keys()}")
        for resource_name, resource in self.extra_resources.items():
            path_for_the_resource = os.path.join(
                model_folder, f"{resource_name}{EXTRA_RESOURCE_APPENDIX}.json"
            )
            with open(path_for_the_resource, "w", encoding="utf-8") as f:
                self.logger.info(
                    f"Saving the extra resource {resource_name} to: {path_for_the_resource}"
                )
                json.dump(resource, f)

    def __store_configuration_info(self, model_folder):
        self.logger.debug(
            f"Saving provided training configuration together with the model checkpoint..."
        )
        path = os.path.join(model_folder, "training_config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.training_config_as_dict, f, indent=4)


@dataclass
class ModelAndResources:
    model: Module
    data_fields: Dict[str, FieldDefinition] = None
    extra_resources: Dict[str, Any] = None
    tokenizer: PreTrainedTokenizer = None

    __annotations__ = {
        "model": Module,
        "data_fields": Dict[str, FieldDefinition],
        "extra_resources": Dict[str, Any],
        "tokenizer": PreTrainedTokenizer,
    }


class ModelLoader:
    logger: ClassVar[logging.Logger] = logging.getLogger(__name__)

    def __init__(self, model_class: Type[PreTrainedModel]):
        self.model_class = model_class

    def load_checkpoint(self, model_path, load_tokenizer=False):
        tokenizer = None
        if load_tokenizer:
            self.logger.info(
                "Loading tokenizer together with the model (assuming that tokenizer data was stored with it)"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            self.logger.info(
                "load_tokenizer=False, you will need to load the corresponding tokenizer externally"
            )
        model = self.model_class.from_pretrained(model_path)

        data_fields = self.__load_data_fields(model_path)
        extra_resources = self.__load_extra_resources(model_path)

        model_and_resources = ModelAndResources(
            model=model,
            data_fields=data_fields,
            extra_resources=extra_resources,
            tokenizer=tokenizer,
        )
        return model_and_resources

    def __load_data_fields(self, model_path):
        data_fields: Dict[str, FieldDefinition] = {}
        stored_fields = [
            filename
            for filename in os.listdir(model_path)
            if FieldDefinition.STORE_NAME_APPENDIX in filename
        ]
        for stored_field_file_name in stored_fields:
            field_name = stored_field_file_name.split(
                FieldDefinition.STORE_NAME_APPENDIX
            )[0]
            if os.path.exists(os.path.join(model_path, stored_field_file_name)):
                stored_field = FieldDefinition.deserialize_from_file(
                    base_path=model_path, field_name=field_name
                )
                data_fields[field_name] = stored_field
            else:
                self.logger.warning(
                    f"Field {field_name} could not be loaded, does it exists?"
                )
        return data_fields

    def __load_extra_resources(self, model_path):
        extra_resources: Dict[str, Any] = {}
        extra_resource_files = [
            filename
            for filename in os.listdir(model_path)
            if EXTRA_RESOURCE_APPENDIX in filename
        ]
        for extra_resource_file in extra_resource_files:
            extra_resource_name = extra_resource_file.split(EXTRA_RESOURCE_APPENDIX)[0]
            resource_path = os.path.join(model_path, extra_resource_file)
            if os.path.exists(os.path.join(model_path, extra_resource_file)):
                with open(resource_path, "r", encoding="utf-8") as f:
                    resource = json.load(f)
                    extra_resources[extra_resource_name] = resource
            else:
                self.logger.warning(
                    f"Extra resource {extra_resource_name} could not be loaded, does it exists?"
                )
        return extra_resources


class ConditionalCheckpointSaver:
    logger: ClassVar[logging.Logger] = logging.getLogger(__name__)

    def __init__(
            self,
            model_saver: ModelSaver,
            conditioning_metric_name: str,
            maximize_metric: bool = True,
            early_stopping_patience: int = None,
            max_checkpoints_to_keep: int = None,
    ):
        self.model_saver = model_saver
        self.conditioning_metric_name = conditioning_metric_name
        self.maximize_metric = maximize_metric
        self.early_stopping_patience = early_stopping_patience
        self.best_value_so_far = None
        self.step_with_best_value = None
        self.patience_exhausted = False

        self.saved_checkpoints_path: List[str] = []
        self.max_checkpoints_to_keep: int = max_checkpoints_to_keep

    def check_and_store(
            self,
            model,
            metrics: Dict[str, float],
            epoch,
            iteration,
            metrics_for_checkpoint: List[BaseEvaluationMetric],
    ):
        metrics_for_checkpoint_names: Set[str] = {
            metric.name for metric in metrics_for_checkpoint
        }
        if self.best_value_so_far is None:
            saved_checkpoint_path = self.model_saver.save_checkpoint(
                model=model,
                epoch=epoch,
                iteration=iteration,
                metrics={
                    m: v
                    for m, v in metrics.items()
                    if m in metrics_for_checkpoint_names
                },
            )
            self.saved_checkpoints_path.append(saved_checkpoint_path)
            self.best_value_so_far = metrics[self.conditioning_metric_name]
            self.step_with_best_value = epoch
        else:
            conditioning_metric_value = metrics[self.conditioning_metric_name]
            metric_has_improved = self.__check_if_metric_has_improved(
                conditioning_metric_value
            )
            if metric_has_improved:
                self.logger.info(
                    f"Better value for {self.conditioning_metric_name},"
                    f" previous was:{self.best_value_so_far} (from epoch:{self.step_with_best_value})"
                    f", new one is: {conditioning_metric_value} at epoch {epoch} (iter {iteration})"
                    f" (metric-diff:{conditioning_metric_value - self.best_value_so_far} ; epoch diff:{epoch - self.step_with_best_value})"
                )
                self.best_value_so_far = conditioning_metric_value
                self.step_with_best_value = epoch
                saved_checkpoint_path = self.model_saver.save_checkpoint(
                    model=model,
                    epoch=epoch,
                    iteration=iteration,
                    metrics={
                        m: v
                        for m, v in metrics.items()
                        if m in metrics_for_checkpoint_names
                    },
                )
                self.saved_checkpoints_path.append(saved_checkpoint_path)
            self.__check_and_remove_oldest_checkpoints()

        if self.early_stopping_patience:
            self.patience_exhausted = self.__check_if_patience_has_been_exhausted(
                current_step=epoch
            )

    def __check_and_remove_oldest_checkpoints(self):
        if (
                self.max_checkpoints_to_keep
                and len(self.saved_checkpoints_path) > self.max_checkpoints_to_keep
        ):
            checkpoints_to_remove = self.saved_checkpoints_path[
                                    : -self.max_checkpoints_to_keep
                                    ]
            checkpoints_to_retain = self.saved_checkpoints_path[
                                    -self.max_checkpoints_to_keep:
                                    ]
            self.logger.info(
                f"Retaining only the latest {self.max_checkpoints_to_keep} checkpoints, removing the older ones..."
            )
            for chpk_to_remove in checkpoints_to_remove:
                self.logger.info(f"Removing old checkpoint: {checkpoints_to_remove}")
                shutil.rmtree(chpk_to_remove, ignore_errors=True)
            self.saved_checkpoints_path = checkpoints_to_retain

    def __check_if_patience_has_been_exhausted(self, current_step):
        return (current_step - self.step_with_best_value) > self.early_stopping_patience

    def __check_if_metric_has_improved(self, conditioning_metric_value: float):
        if self.maximize_metric:
            return conditioning_metric_value > self.best_value_so_far
        else:
            return conditioning_metric_value < self.best_value_so_far

    def is_early_stopping_patience_exhausted(self):
        return self.patience_exhausted


class SequenceLabellingPostProcessor(BaseModelOutcomePostProcessor):
    def __init__(
            self,
            pad_idx: int,
            ctx_len: int,
            num_start_special_tokens: int = 1,
            num_end_special_tokens: int = 1,
    ):
        self.pad_idx: int = pad_idx
        self.start_ctx: int = (ctx_len or 0) + num_start_special_tokens
        self.end_ctx: int = (ctx_len or 0) + num_end_special_tokens
        self.total_ctx_len: int = self.start_ctx + self.end_ctx
        self.num_start_special_tokens = num_start_special_tokens
        self.num_end_special_tokens = num_end_special_tokens

    def process(
            self, gold_labels: Tensor, model_outcomes: Tensor
    ):
        golds: Tensor = gold_labels.cuda()
        logits = model_outcomes.cuda()
        multilabel_setting = len(gold_labels.shape) == 3
        outcomes = (
            torch.sigmoid(logits) if multilabel_setting else torch.max(logits, dim=2)[1]
        )

        golds = torch.narrow(
            golds,
            dim=1,
            start=self.start_ctx,
            length=golds.shape[1] - self.total_ctx_len,
        )
        outcomes = torch.narrow(
            outcomes,
            dim=1,
            start=self.start_ctx,
            length=outcomes.shape[1] - self.total_ctx_len,
        )

        if multilabel_setting:
            golds = golds.reshape(-1, golds.shape[2])
            outcomes = outcomes.reshape(-1, outcomes.shape[2])
            non_padding_mask = golds[:, self.pad_idx].reshape(-1).ne(1.0)
            non_padding_mask = non_padding_mask.unsqueeze(1).expand(-1, golds.shape[1])
            golds = torch.masked_select(golds, mask=non_padding_mask).reshape(
                -1, golds.shape[1]
            )
            outcomes = torch.masked_select(outcomes, mask=non_padding_mask).reshape(
                -1, outcomes.shape[1]
            )
        else:
            non_padding_mask = golds.reshape(-1).ne(self.pad_idx)
            golds = torch.masked_select(golds.reshape(-1), mask=non_padding_mask)
            outcomes = torch.masked_select(outcomes.reshape(-1), mask=non_padding_mask)
        return golds, outcomes


class TextCleaner:
    REPLACEMENTS: ClassVar[Dict[str, str]] = {
        "’": "'",
        "‘": "'",
        "ʹ": "'",
        "‑": "-",
        "–": "-",
        "—": "-",
        "…": "...",
        "“": '"',
        "”": '"',
        "ˮ": '"',
    }

    def clean_text(self, text: str):
        cleaned_text = []
        for c in text:
            cleaned_text.append(self.REPLACEMENTS.get(c, c))
        return "".join(cleaned_text)


class MapaEntityDetectionTrainerConfig(BaseModel):
    model_name: str = Field(..., description="The name of the trained model")
    model_version: int = Field(..., description="The version number of the model")
    model_description: Optional[str] = Field(
        description="An optional description of the model"
    )

    pretrained_model_name_or_path: Union[str] = Field(
        ..., description="The name or path to the pretrained base model"
    )
    pretrained_tokenizer_name_or_path: Optional[str] = Field(
        default=None,
        description="Name/path to pretrained tokenizer, "
                    "only in case it differs from the model name/path",
    )
    do_lower_case: bool = Field(
        default=False, description="Force the tokenizer's do_lower_case to True"
    )

    train_set: str = Field(..., description="Name of the training data file")
    dev_set: str = Field(
        ..., description="Name of the validation (a.k.a dev) data file"
    )

    checkpoints_folder: Union[Path, str] = Field(
        ..., description="Directory to store the model checkpoints during the training"
    )
    max_checkpoints_to_keep: Optional[int] = Field(
        None,
        description="Number of checkpoints to keep during training (the oldest ones are removed)",
    )

    num_epochs: int = Field(200, description="The maximum number of epochs to train")
    early_stopping_patience: int = Field(
        50,
        description="Number of epochs without improvement before early-stopping the training",
    )

    batch_size: int = Field(
        ...,
        description="The batch size (number of examples processed at once) during the training",
    )

    train_iters_to_eval: int = Field(
        default=-1,
        description="The number of iterations before triggering an evaluation (apart from full epochs)",
    )

    gradients_accumulation_steps: int = Field(
        default=1,
        description="The number of steps to accumulate gradients before an optimizer step",
    )
    clip_grad_norm: float = Field(
        default=1.0, description="The value for gradient clipping"
    )

    lr: float = Field(2e-05, description="The learning rate for the training")
    warmup_epochs: float = Field(
        default=1.0,
        description="The number of epochs to warmup the learning rate. Admits non-integer values, e.g. 0.5",
    )

    amp: bool = Field(
        default=True,
        description="The use of Automatic Mixed Precision (amp) to leverage fp16 operations and speed-up GPU training",
    )

    random_seed: int = Field(
        default=42, description="The random seed, to make the experiments reproducible"
    )

    valid_seq_len: int = Field(
        default=300,
        description="The length of the sliding-window (the central valid part excluding the contexts)",
    )
    ctx_len: int = Field(
        default=100,
        description="The length of the context surrounding each sliding-window",
    )
    labels_to_omit: str = Field(
        default="O,X,[CLS],[SEP],[PAD]",
        description="The labels omitted from the in-training evaluation (default for BERT)",
    )

    def pretty_print_config(self):
        params = []
        for arg in vars(self):
            params.append(f"{arg}: {getattr(self, arg)}")
        return (
                "\n"
                "==================================================\n"
                "  Experiment configuration and hyper-parameters:"
                "\n"
                "==================================================\n  "
                + "\n  ".join(params)
                + "\n================================================"
        )


class EnhancedTwoFlatLevelsSequenceLabellingConfig(BertConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_level2_labels = None
        self.num_level1_labels = None


class TokenPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class EnhancedTwoFlatLevelsSequenceLabellingModel(BertPreTrainedModel):
    def __init__(self, config: EnhancedTwoFlatLevelsSequenceLabellingConfig):
        super().__init__(config)
        self.num_level1_labels = config.num_level1_labels
        self.num_level2_labels = config.num_level2_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.level1_labels_pooler = TokenPooler(config)
        self.level2_labels_pooler = TokenPooler(config)
        self.level1_labels_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.level2_labels_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.level1_labels_classifier = nn.Linear(
            config.hidden_size, self.num_level1_labels
        )
        self.level2_labels_classifier = nn.Linear(
            config.hidden_size, self.num_level2_labels
        )

        self.level1_loss_fct = CrossEntropyLoss()
        self.level2_loss_fct = CrossEntropyLoss()

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            ctx_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            level1_labels=None,
            level2_labels=None,
            output_attentions=None,
            output_hidden_states=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]

        level1_labels_pooled = self.level1_labels_pooler(sequence_output)
        level1_labels_pooled = self.level1_labels_dropout(level1_labels_pooled)
        level1_labels_logits = self.level1_labels_classifier(level1_labels_pooled)

        level2_labels_pooled = self.level2_labels_pooler(sequence_output)
        level2_labels_pooled = self.level2_labels_dropout(level2_labels_pooled)
        level2_labels_logits = self.level2_labels_classifier(level2_labels_pooled)

        loss = None
        if level1_labels is not None:
            if attention_mask is not None:
                active_loss = (attention_mask.view(-1) * ctx_mask.view(-1)) == 1
                active_level1_labels_logits = level1_labels_logits.view(
                    -1, self.num_level1_labels
                )
                active_level2_labels_logits = level2_labels_logits.view(
                    -1, self.num_level2_labels
                )
                active_level1_labels = torch.where(
                    active_loss,
                    level1_labels.view(-1),
                    torch.tensor(self.level1_loss_fct.ignore_index).type_as(
                        level1_labels
                    ),
                )
                active_level2_labels = torch.where(
                    active_loss,
                    level2_labels.view(-1),
                    torch.tensor(self.level2_loss_fct.ignore_index).type_as(
                        level2_labels
                    ),
                )
                level1_labels_loss = self.level1_loss_fct(
                    active_level1_labels_logits, active_level1_labels
                )
                level2_labels_loss = self.level2_loss_fct(
                    active_level2_labels_logits, active_level2_labels
                )
            else:
                level1_labels_loss = self.level1_loss_fct(
                    level1_labels_logits.view(-1, self.num_level1_labels),
                    level1_labels.view(-1),
                )
                level2_labels_loss = self.level2_loss_fct(
                    level2_labels_logits.view(-1, self.num_level2_labels),
                    level2_labels.view(-1),
                )

            loss = level1_labels_loss + level2_labels_loss

        output = (
            level1_labels_logits,
            level2_labels_logits,
        )
        return ((loss,) + output) if loss is not None else output

    def _reorder_cache(self, past, beam_idx):
        pass


class WindowedSequencesGenerator:
    INSTANCE_IDX: ClassVar[str] = "idx"
    INSTANCE_ORDER: ClassVar[str] = "order"

    def __init__(
            self,
            valid_seq_len: int,
            ctx_len: int,
            fields_to_process: List[FieldDefinition],
            add_cls_sep: bool,
    ):
        self.valid_seq_len = valid_seq_len
        self.ctx_len = ctx_len
        self.fields_to_process = fields_to_process
        self.add_cls_sep = add_cls_sep

    def transform_sequences_to_windows_with_ctx(
            self, instances: List[Dict[str, List[str]]]
    ):
        converted_instances: List[Dict[str, List[str]]] = []
        logger.info(f"Converting instances to windowed sequences with context...")
        for instance_num, instance in enumerate(tqdm(instances)):
            converted_instances += self.__transform_instance(instance_num, instance)
        return converted_instances

    def __transform_instance(
            self, instance_num, instance: Dict[str, List[str]]
    ):
        resulting_windowed_instances: List[Dict[str, List[str]]] = []
        for field in self.fields_to_process:
            windows_with_ctx = self.__obtain_windows_with_context(
                instance[field.name], field=field
            )
            for window_num, window_with_ctx in enumerate(windows_with_ctx):
                if self.add_cls_sep:
                    window_with_ctx = self.__add_special_tokens(
                        window_with_ctx, field=field
                    )
                if len(resulting_windowed_instances) == window_num:
                    resulting_windowed_instances.append(
                        {
                            self.INSTANCE_IDX: instance_num,
                            self.INSTANCE_ORDER: window_num,
                            field.name: window_with_ctx,
                        }
                    )
                else:
                    resulting_windowed_instances[window_num].update(
                        {field.name: window_with_ctx}
                    )
        return resulting_windowed_instances

    def __obtain_windows_with_context(
            self, original_sequence: List[str], field: FieldDefinition
    ):
        windows: List[List[str]] = []
        for i in range(0, len(original_sequence), self.valid_seq_len):
            windows.append(original_sequence[i: i + self.valid_seq_len])
        windows_with_ctx: List[List[str]] = self.__add_ctx_windows(
            windows=windows, field=field
        )
        return windows_with_ctx

    def __add_ctx_windows(
            self, windows: List[List[str]], field: FieldDefinition
    ):
        windows_with_ctx = []
        for window_num, window in enumerate(windows):
            prev_ctx = self.__obtain_prev_ctx(
                window_num=window_num, windows=windows, field=field
            )
            post_ctx = self.__obtain_post_ctx(
                window_num=window_num, windows=windows, field=field
            )
            window += [field.pad_token] * (self.valid_seq_len - len(window))
            window_with_ctx = prev_ctx + window + post_ctx
            assert len(window_with_ctx) == 2 * self.ctx_len + len(window)
            windows_with_ctx.append(window_with_ctx)
        return windows_with_ctx

    def __obtain_prev_ctx(
            self, window_num, windows, field: FieldDefinition
    ):
        n = min(ceil(self.ctx_len / self.valid_seq_len), window_num)
        prev_ctx: List[str] = list(
            itertools.chain.from_iterable(windows[window_num - n: window_num])
        )[-self.ctx_len:]
        prev_ctx = [field.pad_token] * (self.ctx_len - len(prev_ctx)) + prev_ctx
        assert len(prev_ctx) == self.ctx_len
        return prev_ctx

    def __obtain_post_ctx(
            self, window_num, windows, field: FieldDefinition
    ):
        n = min(ceil(self.ctx_len / self.valid_seq_len), len(windows))
        post_ctx = list(
            itertools.chain.from_iterable(windows[window_num + 1: window_num + n + 1])
        )[: self.ctx_len]
        post_ctx = post_ctx + [field.pad_token] * (self.ctx_len - len(post_ctx))
        assert len(post_ctx) == self.ctx_len
        return post_ctx

    @classmethod
    def __find_first_non_padding_position(cls, seq: List[str], field: FieldDefinition):
        for i, elem in enumerate(seq):
            if elem != field.pad_token:
                return i

    @classmethod
    def __find_last_non_padding_position(cls, seq: List[str], field: FieldDefinition):
        for i, elem in enumerate(seq[::-1]):
            if elem != field.pad_token:
                return len(seq) - i

    def __add_special_tokens(self, window_with_ctx: List[str], field: FieldDefinition):
        ww_ctx = window_with_ctx
        first_npp = self.__find_first_non_padding_position(window_with_ctx, field=field)
        last_npp = self.__find_last_non_padding_position(window_with_ctx, field=field)
        window_with_ctx = (
                ww_ctx[:first_npp]
                + [field.cls_token]
                + ww_ctx[first_npp:last_npp]
                + [field.sep_token]
                + ww_ctx[last_npp:]
        )
        return window_with_ctx

    @classmethod
    def __calculate_seq_split_points(cls, seq_indices: List[int]):
        current_index = seq_indices[0]
        seq_split_points = []
        for i, seq_idx in enumerate(seq_indices):
            if seq_idx != current_index:
                seq_split_points.append(i)
                current_index = seq_idx
        seq_split_points.append(len(seq_indices))
        return seq_split_points


CharSpan = Tuple[int, int]
TokenList = List[str]


class TokenizationRemappingHelper:
    def __init__(
            self,
            pretokenized_instances: List[Dict[str, List[str]]],
            tokenizer: PreTrainedTokenizer,
            tokens_field_name: str,
            tag_fields_names: Tuple[str, ...],
            subword_tag: str = "X",
    ):
        self.tokenizer = tokenizer
        self.tokens_field_name = tokens_field_name
        self.tag_fields_names = tag_fields_names
        self.original_tokens: List[List[str]] = [
            instance[tokens_field_name] for instance in pretokenized_instances
        ]
        self.subword_tag = subword_tag
        (
            self.retokenized_instances,
            self.orig_to_tok_mappings,
        ) = self.__retokenize_and_remap(pretokenized_instances)

    def __retokenize_and_remap(
            self, pretokenized_instances: List[Dict[str, List[str]]]
    ):
        retokenized_instances = []
        all_instances_orig_to_tok_map = []
        logger.info(
            "Re-tokenizing using the chosen Transformers tokenizer and calculating labels remapping..."
        )
        for instance in tqdm(pretokenized_instances):
            orig_tokens: List[str] = instance[self.tokens_field_name]
            orig_tags_collection: List[List[str]] = [
                instance[tags_f_name] for tags_f_name in self.tag_fields_names
            ]
            target_tokens: List[str] = []
            target_tags_collection: List[List[str]] = [
                [] for _ in range(len(orig_tags_collection))
            ]
            orig_to_tok_map = []
            for i, token in enumerate(orig_tokens):
                orig_to_tok_map.append(len(target_tokens))
                current_target_tokens = self.tokenizer.tokenize(token)
                if len(current_target_tokens) == 0:
                    continue
                target_tokens += current_target_tokens
                for orig_tags_num, orig_tags in enumerate(orig_tags_collection):
                    target_tags_collection[orig_tags_num] += [orig_tags[i]] + [
                        self.subword_tag
                    ] * (len(current_target_tokens) - 1)

            retokenized_instances.append(
                {
                    self.tokens_field_name: target_tokens,
                    **{
                        tags_f_name: target_tags_collection[num]
                        for num, tags_f_name in enumerate(self.tag_fields_names)
                    },
                }
            )
            all_instances_orig_to_tok_map.append(orig_to_tok_map)
        return retokenized_instances, all_instances_orig_to_tok_map


label_misspellings: Tuple[Tuple[str, str], ...] = (("ORGANIZATION", "ORGANISATION"),)


class MapaEntitiesHierarchy:
    def __init__(
            self,
            hierarchy_dict: Dict[str, Set[str]],
            label_misspellings: Tuple[Tuple[str, str], ...] = label_misspellings,
    ):
        self.__hierarchy_dict: OrderedDict[str, List[str]] = OrderedDict()
        for level1_label in sorted(hierarchy_dict.keys()):
            self.__hierarchy_dict[level1_label] = sorted(hierarchy_dict[level1_label])
        self.__reversed_hierarchy = {
            v: k for k, values in self.__hierarchy_dict.items() for v in values
        }
        self.__capitalization_correction_map = {
            **{k.lower(): k for k in self.__hierarchy_dict.keys()},
            **{
                value.lower(): value
                for values in list(self.__hierarchy_dict.values())
                for value in values
            },
        }
        self.label_mispellings_map: Dict[str, str] = {
            incorrect: correct for incorrect, correct in label_misspellings
        }

    def get_level1_labels(self, bio: bool, lowercase: bool):
        labels: List[str] = list(self.__hierarchy_dict.keys())
        if lowercase:
            labels = [label.lower() for label in labels]
        if bio:
            labels = [f"B-{label}" for label in labels] + [
                f"I-{label}" for label in labels
            ]
        return labels

    def get_all_level2_labels(self, bio: bool, lowercase: bool):
        all_level2_labels: List[str] = sorted(self.__reversed_hierarchy.keys())
        if lowercase:
            all_level2_labels = [label.lower() for label in all_level2_labels]
        if bio:
            all_level2_labels = [f"B-{label}" for label in all_level2_labels] + [
                f"I-{label}" for label in all_level2_labels
            ]
        return all_level2_labels

    def to_json(self):
        return self.__hierarchy_dict


class TwoFlatLevelsDataset(Dataset):
    O_TAG: ClassVar[str] = "O"

    def __init__(
            self,
            instances: List[Dict[str, List[str]]],
            tokenizer: PreTrainedTokenizer,
            valid_seq_len: int,
            ctx_len: int,
            input_field: FieldDefinition,
            level1_tags_field: FieldDefinition,
            level2_tags_field: FieldDefinition,
            build_vocabs: bool,
            entities_hierarchy: Optional[MapaEntitiesHierarchy],
            train_subwords: bool = True,
            loss_fct_ignore_index: int = -100,
            linebreak_token: str = None,
    ):
        absolute_max_len = 512 - 2
        if (2 * ctx_len + valid_seq_len) > absolute_max_len:
            raise Exception(
                f"Valid sequence len ({valid_seq_len}) and ctx_len ({ctx_len}) exceed the absolute max len of {absolute_max_len}"
            )

        self.original_instances = instances

        self.__clean_tokens_(
            instances=self.original_instances,
            tokens_field_name=input_field.name,
            text_cleaner=TextCleaner(),
        )

        self.tokenizer = tokenizer
        if linebreak_token:
            self.__add_linebreak_token_to_tokenizer_(
                self.tokenizer, linebreak_token=linebreak_token
            )
        self.valid_seq_len = valid_seq_len
        self.ctx_len = ctx_len
        self.input_field = input_field
        self.level1_tags_field = level1_tags_field
        self.level2_tags_field = level2_tags_field
        self.train_subwords = train_subwords
        self.loss_fct_ignore_index = loss_fct_ignore_index
        self.tokens_remapper = TokenizationRemappingHelper(
            self.original_instances,
            tokenizer=self.tokenizer,
            tokens_field_name=input_field.name,
            tag_fields_names=(
                level1_tags_field.name,
                level2_tags_field.name,
            ),
        )

        if build_vocabs:
            level1_labels: List[str] = entities_hierarchy.get_level1_labels(
                bio=True, lowercase=False
            )
            level2_labels: List[str] = entities_hierarchy.get_all_level2_labels(
                bio=True, lowercase=False
            )
            if self.train_subwords:
                level1_labels = [self.tokens_remapper.subword_tag] + level1_labels
                level2_labels = [self.tokens_remapper.subword_tag] + level2_labels
            self.level1_tags_field.build_labels_vocab(direct_vocab=level1_labels)
            self.level2_tags_field.build_labels_vocab(direct_vocab=level2_labels)

        windowed_sequences_generator = WindowedSequencesGenerator(
            valid_seq_len=valid_seq_len,
            ctx_len=ctx_len,
            fields_to_process=[
                self.input_field,
                self.level1_tags_field,
                self.level2_tags_field,
            ],
            add_cls_sep=True,
        )
        self.windowed_instances = (
            windowed_sequences_generator.transform_sequences_to_windows_with_ctx(
                self.tokens_remapper.retokenized_instances
            )
        )

    @classmethod
    def __add_linebreak_token_to_tokenizer_(
            cls, tokenizer: PreTrainedTokenizer, linebreak_token: str
    ):
        tokenizer.add_tokens([linebreak_token])

    @classmethod
    def __clean_tokens_(
            cls,
            instances: List[Dict[str, List[str]]],
            tokens_field_name,
            text_cleaner: TextCleaner,
    ):
        for instance in instances:
            instance[tokens_field_name] = list(instance[tokens_field_name])
            for i, token in enumerate(instance[tokens_field_name]):
                instance[tokens_field_name][i] = text_cleaner.clean_text(token)

    @classmethod
    def load_from_file(
            cls,
            input_path,
            tokenizer: PreTrainedTokenizer,
            valid_seq_len: int,
            ctx_len: int,
            input_field: FieldDefinition,
            level1_tags_field: FieldDefinition,
            level2_tags_field: FieldDefinition,
            build_vocabs: bool,
            entities_hierarchy: Optional[MapaEntitiesHierarchy],
            train_subwords: bool,
    ):
        with open(input_path, "r", encoding="utf-8") as f:
            instances = [json.loads(line) for line in f.readlines()]
        return TwoFlatLevelsDataset(
            instances,
            tokenizer,
            valid_seq_len,
            ctx_len,
            input_field,
            level1_tags_field,
            level2_tags_field,
            build_vocabs,
            entities_hierarchy,
            train_subwords=train_subwords,
        )

    def __len__(self):
        return len(self.windowed_instances)

    def __getitem__(self, item):
        return self.__process_instance(self.windowed_instances[item])

    def __process_instance(self, instance):
        instance_num = instance[WindowedSequencesGenerator.INSTANCE_IDX]
        instance_order = instance[WindowedSequencesGenerator.INSTANCE_ORDER]

        tokens = instance[self.input_field.name]
        level1_tags = instance[self.level1_tags_field.name]
        level2_tags = instance[self.level2_tags_field.name]

        input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens))
        attention_mask = torch.ne(input_ids, self.tokenizer.pad_token_id).long()
        ctx_mask = torch.tensor(
            [0] * (self.ctx_len + 1)
            + [1] * self.valid_seq_len
            + [0] * (self.ctx_len + 1)
        )
        if self.train_subwords:
            level1_tags_ids = torch.tensor(
                [self.level1_tags_field.stoi(label) for label in level1_tags]
            )
            level2_tags_ids = torch.tensor(
                [self.level2_tags_field.stoi(label) for label in level2_tags]
            )
        else:
            level1_tags_ids = torch.tensor(
                [
                    self.level1_tags_field.stoi(label)
                    if label != self.tokens_remapper.subword_tag
                    else self.loss_fct_ignore_index
                    for label in level1_tags
                ]
            )
            level2_tags_ids = torch.tensor(
                [
                    self.level2_tags_field.stoi(label)
                    if label != self.tokens_remapper.subword_tag
                    else self.loss_fct_ignore_index
                    for label in level2_tags
                ]
            )

        processed_instance = {
            WindowedSequencesGenerator.INSTANCE_IDX: torch.tensor(
                instance_num, dtype=torch.long
            ),
            WindowedSequencesGenerator.INSTANCE_ORDER: torch.tensor(
                instance_order, dtype=torch.long
            ),
            self.input_field.name: input_ids,
            self.input_field.attn_mask_name(): attention_mask,
            self.input_field.name + "_ctx_mask": ctx_mask,
            self.level1_tags_field.name: level1_tags_ids,
            self.level2_tags_field.name: level2_tags_ids,
        }
        return processed_instance


class MapaEntityDetectionTrainer:
    def __init__(self, config: MapaEntityDetectionTrainerConfig, mapa_entites_hierarchy: MapaEntitiesHierarchy):
        preinstantiation_stuff()
        self.cfg: MapaEntityDetectionTrainerConfig = config
        logger.info(self.cfg.pretty_print_config())
        if self.cfg.random_seed and self.cfg.random_seed > 0:
            set_all_random_seeds_to(self.cfg.random_seed)
        else:
            logger.warning(
                "No random seed has been configured. The training will be non-deterministic."
            )

        self.tokenizer = self.load_tokenizer(self.cfg)
        self.tokens_field = FieldDefinition.create_sequential_field(
            "tokens", multilabel=False, tokenizer=self.tokenizer
        )
        self.level1_tags_field = FieldDefinition.create_sequential_field(
            "level1_tags", multilabel=False, tokenizer=self.tokenizer, default_label="O"
        )
        self.level2_tags_field = FieldDefinition.create_sequential_field(
            "level2_tags", multilabel=False, tokenizer=self.tokenizer, default_label="O"
        )

        datasets_vocabs_and_extra_resources: DatasetsVocabsAndExtraResources = (
            self.define_fields_datasets_and_resources(
                config=self.cfg,
                tokenizer=self.tokenizer,
                mapa_entites_hierarchy=mapa_entites_hierarchy
            )
        )
        self.train_dataset: Dataset = datasets_vocabs_and_extra_resources.train_dataset
        self.dev_dataset: Dataset = datasets_vocabs_and_extra_resources.dev_dataset
        self.data_fields: Dict[
            str, FieldDefinition
        ] = self.__create_data_fields_inventory(
            datasets_vocabs_and_extra_resources.data_fields
        )
        self.extra_resources: Dict[
            str, Any
        ] = datasets_vocabs_and_extra_resources.extra_resources

        self.model = self.instantiate_model(self.cfg)
        self.use_amp = self.cfg.amp and torch.cuda.is_available()
        self.model = to_data_parallel(self.model, use_amp=self.use_amp)
        self.is_multigpu_setting = (
                torch.cuda.is_available() and torch.cuda.device_count() > 1
        )

        self.optimizer = self.instantiate_optimizer(config=self.cfg, model=self.model)
        self.lr_scheduler = self.instantiate_lr_scheduler(
            optimizer=self.optimizer,
            config=self.cfg,
            num_training_instances=len(self.train_dataset),
        )

        self.metrics_holder: EvaluationMetricsHolder = (
            self.instantiate_evaluation_metrics(self.cfg)
        )
        self.train_evaluation_metrics_accumulator = (
            self.instantiate_evaluation_metric_accumulator(
                evaluation_metrics_holder=self.metrics_holder
            )
        )
        self.dev_evaluation_metrics_accumulator = (
            self.instantiate_evaluation_metric_accumulator(
                evaluation_metrics_holder=self.metrics_holder
            )
        )

        self.conditional_checkpoint_saver = (
            self.instantiate_conditional_checkpoint_saver(
                self.cfg, tokenizer=self.tokenizer
            )
        )
        self.gradients_accumulation_steps = self.cfg.gradients_accumulation_steps

        self.scaler = GradScaler(enabled=self.use_amp)

    @classmethod
    def load_tokenizer(
            cls, config: MapaEntityDetectionTrainerConfig
    ):
        do_lower_case = (
                "uncased" in config.pretrained_model_name_or_path or config.do_lower_case
        )

        tokenizer_name_or_path = config.pretrained_model_name_or_path
        if config.pretrained_tokenizer_name_or_path:
            logger.info(
                f"Tokenizer will be loaded from specified name/path: {config.pretrained_tokenizer_name_or_path}"
            )
            tokenizer_name_or_path = config.pretrained_tokenizer_name_or_path
        else:
            logger.info(
                f"Tokenizer will be loaded from the same name/path than the pre-trained model"
            )

        logger.info(
            f"Loading pre-trained TOKENIZER: >> {tokenizer_name_or_path} << (do_lower_case={do_lower_case})"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=tokenizer_name_or_path,
            do_lower_case=do_lower_case,
            cache_dir=os.path.join("core", "training", "cache"),
            output_loading_info=True,
        )
        return tokenizer

    def instantiate_optimizer(
            self, config: MapaEntityDetectionTrainerConfig, model: Module
    ):
        logger.info(f"Instantiating and AdamW optimizer")
        return instantiate_adamw(model=model, lr=config.lr, correct_bias=False)

    def instantiate_lr_scheduler(
            self,
            optimizer: Optimizer,
            config: MapaEntityDetectionTrainerConfig,
            num_training_instances: int,
    ):
        if config.warmup_epochs and config.warmup_epochs > 0:
            logger.info(
                f"Instantiating LR Scheduler with linear warmup of {config.warmup_epochs} epochs"
            )
            return instantiate_linear_scheduler_with_warmup(
                optimizer=optimizer,
                num_training_instances=num_training_instances,
                config=config,
            )
        else:
            logger.info(f"Instantiating constant LR Scheduler")
            return instantiate_constant_lr_scheduler(optimizer=optimizer)

    @classmethod
    def instantiate_evaluation_metric_accumulator(
            cls, evaluation_metrics_holder: EvaluationMetricsHolder
    ):
        return SimpleEvaluationMetricsAccumulator(
            evaluation_metrics_holder.evaluation_metrics
        )

    def instantiate_conditional_checkpoint_saver(
            self, config: MapaEntityDetectionTrainerConfig, tokenizer: PreTrainedTokenizer
    ):
        model_saver = ModelSaver(
            base_folder=config.checkpoints_folder,
            model_name=f"{config.model_name}",
            model_version=config.model_version,
            data_fields=self.data_fields,
            extra_resources=self.extra_resources,
            tokenizer=tokenizer,
            training_config=config,
        )
        return ConditionalCheckpointSaver(
            model_saver=model_saver,
            conditioning_metric_name=self.metrics_holder.metric_to_check.name,
            maximize_metric=self.metrics_holder.maximize_checked_metric,
            early_stopping_patience=config.early_stopping_patience,
            max_checkpoints_to_keep=config.max_checkpoints_to_keep,
        )

    def __create_data_fields_inventory(
            self, data_fields: List[FieldDefinition]
    ):
        all_data_fields_map: Dict[str, FieldDefinition] = {}
        for data_field in data_fields:
            self.__setattr__(data_field.name, data_field)
            all_data_fields_map[data_field.name] = data_field
        return all_data_fields_map

    def define_fields_datasets_and_resources(
            self,
            config: MapaEntityDetectionTrainerConfig,
            tokenizer: PreTrainedTokenizer,
            mapa_entites_hierarchy: MapaEntitiesHierarchy
    ):
        mapa_entities = mapa_entites_hierarchy

        train_set_path = os.path.join("core", "training", config.train_set)
        train_dataset = TwoFlatLevelsDataset.load_from_file(
            input_path=train_set_path,
            tokenizer=tokenizer,
            input_field=self.tokens_field,
            level1_tags_field=self.level1_tags_field,
            level2_tags_field=self.level2_tags_field,
            valid_seq_len=config.valid_seq_len,
            ctx_len=config.ctx_len,
            build_vocabs=True,
            entities_hierarchy=mapa_entities,
            train_subwords=True,
        )

        dev_set_path = os.path.join("core", "training", config.dev_set)
        dev_dataset = TwoFlatLevelsDataset.load_from_file(
            input_path=dev_set_path,
            tokenizer=tokenizer,
            input_field=self.tokens_field,
            level1_tags_field=self.level1_tags_field,
            level2_tags_field=self.level2_tags_field,
            valid_seq_len=config.valid_seq_len,
            ctx_len=config.ctx_len,
            build_vocabs=False,
            entities_hierarchy=None,
            train_subwords=True,
        )

        dataset_vocabs_and_extras = DatasetsVocabsAndExtraResources(
            data_fields=[
                self.tokens_field,
                self.level1_tags_field,
                self.level2_tags_field,
            ],
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            extra_resources={"mapa_entities": mapa_entities.to_json()},
        )
        return dataset_vocabs_and_extras

    def instantiate_model(
            self, config: MapaEntityDetectionTrainerConfig
    ):
        model_config = EnhancedTwoFlatLevelsSequenceLabellingConfig.from_pretrained(
            config.pretrained_model_name_or_path, cache_dir=os.path.join("core", "training", "cache")
        )
        model_config.num_level1_labels = len(self.level1_tags_field.vocab)
        model_config.num_level2_labels = len(self.level2_tags_field.vocab)

        print(
            f"Loading pre-trained BERT with (enhanced) "
            f"two-levels token classification for MAPA with {model_config.num_level1_labels} and {model_config.num_level2_labels}"
            f" classes for each level respectively"
        )
        model = EnhancedTwoFlatLevelsSequenceLabellingModel.from_pretrained(
            config.pretrained_model_name_or_path,
            cache_dir=os.path.join("core", "training", "cache"),
            config=model_config,
        )
        model.resize_token_embeddings(len(self.tokenizer))

        return model

    def model_step(
            self,
            config: MapaEntityDetectionTrainerConfig,
            model: Module,
            batch,
            batch_number: int,
    ):
        inputs = batch[self.tokens_field.name]
        ctx_mask = batch[self.tokens_field.name + "_ctx_mask"]
        attention_mask = batch[self.tokens_field.attn_mask_name()]
        level1_labels = batch[self.level1_tags_field.name]
        level2_labels = batch[self.level2_tags_field.name]

        outputs = model(
            inputs,
            attention_mask=attention_mask,
            ctx_mask=ctx_mask,
            level1_labels=level1_labels,
            level2_labels=level2_labels,
        )
        loss, level1_labels_logits, level2_labels_logits = outputs[:3]
        model_step_output = ModelStepOutput(
            loss=loss,
            prediction_scores={
                self.level1_tags_field.name: level1_labels_logits,
                self.level2_tags_field.name: level2_labels_logits,
            },
            gold_labels={
                self.level1_tags_field.name: level1_labels,
                self.level2_tags_field.name: level2_labels,
            },
        )
        return model_step_output

    def instantiate_evaluation_metrics(
            self, config: MapaEntityDetectionTrainerConfig
    ):
        level1_labels_to_eval = [
            self.level1_tags_field.stoi(label)
            for label in self.level1_tags_field.vocab.keys()
            if label not in config.labels_to_omit.split(",")
        ]
        level2_labels_to_eval = [
            self.level2_tags_field.stoi(label)
            for label in self.level2_tags_field.vocab.keys()
            if label not in config.labels_to_omit.split(",")
        ]

        level1_binarization_negative_labels = [
                                                  self.level1_tags_field.stoi(tag) for tag in
                                                  config.labels_to_omit.split(",")
                                              ] + [-100]

        level1_microf1 = FscoreMetric(
            name="L1_miF",
            target_field=self.level1_tags_field,
            preprocessor=SequenceLabellingPostProcessor(
                pad_idx=self.level1_tags_field.pad_idx(), ctx_len=config.ctx_len
            ),
            labels_to_evaluate=level1_labels_to_eval,
            average_method="micro",
        )

        level1_binf1 = FscoreMetric(
            name="L1_binF",
            target_field=self.level1_tags_field,
            preprocessor=SequenceLabellingPostProcessor(
                pad_idx=self.level1_tags_field.pad_idx(), ctx_len=config.ctx_len
            ),
            binarization_negative_labels=level1_binarization_negative_labels,
            average_method="binary",
        )

        level2_microf1 = FscoreMetric(
            name="L2_miF",
            target_field=self.level2_tags_field,
            preprocessor=SequenceLabellingPostProcessor(
                pad_idx=self.level2_tags_field.pad_idx(), ctx_len=config.ctx_len
            ),
            labels_to_evaluate=level2_labels_to_eval,
            average_method="micro",
        )

        l1_l2_f1_comb = EvaluationMetricCombination(
            name="all_metrics_comb", metrics_to_combine=(level1_microf1, level2_microf1)
        )

        evaluation_metrics_holder = EvaluationMetricsHolder(
            evaluation_metrics=[
                level1_microf1,
                level1_binf1,
                level2_microf1,
                l1_l2_f1_comb,
            ],
            metric_to_check=l1_l2_f1_comb,
            maximize_checked_metric=True,
            metrics_in_progress_bar=[level1_microf1, level1_binf1, level2_microf1],
            metrics_in_checkpoint_name=[level1_microf1, level1_binf1, level2_microf1],
        )
        return evaluation_metrics_holder

    def __load_train_dataloader(self):
        train_dataloader = DataLoader(
            dataset=self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True
        )
        return train_dataloader

    def __load_dev_dataloader(self):
        dev_dataloader = DataLoader(
            dataset=self.dev_dataset, batch_size=self.cfg.batch_size, shuffle=False
        )
        return dev_dataloader

    def __iteration_based_evaluation_active(self):
        return self.cfg.train_iters_to_eval > 0

    def __should_evaluate(self, current_training_iter: int, full_epoch_finished: bool):
        if self.__iteration_based_evaluation_active():
            return current_training_iter % self.cfg.train_iters_to_eval == 0
        else:
            return full_epoch_finished

    def train(self):
        train_dataloader = self.__load_train_dataloader()
        dev_dataloader = self.__load_dev_dataloader()
        global_iter_number = 0
        for epoch in tqdm(range(self.cfg.num_epochs), ascii=" ||||", position=0):
            global_iter_number = self.__do_train_epoch(
                train_dataloader=train_dataloader,
                dev_dataloader=dev_dataloader,
                epoch=epoch,
                training_iteration=global_iter_number,
            )

            if self.__should_evaluate(
                    current_training_iter=global_iter_number, full_epoch_finished=True
            ):
                self.__evaluation_and_saving(
                    dev_dataloader=dev_dataloader,
                    epoch=epoch,
                    iteration=global_iter_number,
                )
            if self.conditional_checkpoint_saver.is_early_stopping_patience_exhausted():
                logger.info(
                    f" >> Early stopping patience of {self.cfg.early_stopping_patience} epochs exhausted. Stopping the training."
                )
                break

    def __do_train_epoch(
            self,
            train_dataloader: DataLoader,
            dev_dataloader: DataLoader,
            epoch: int,
            training_iteration: int,
    ):
        with batch_progress_bar(
                epoch=epoch, num_epochs=self.cfg.num_epochs, dataloader=train_dataloader
        ) as t:
            for batch_number, batch in enumerate(t):
                metric_values = self.__train_step(
                    batch_number, batch, epoch, iteration=training_iteration
                )
                report_to_progress_bar(
                    progress_bar=t,
                    metric_values=metric_values,
                    metrics_for_progress_bar=self.metrics_holder.metrics_in_progress_bar,
                    current_learning_rate=get_learning_rate(self.optimizer),
                )

                training_iteration += 1
                if self.__should_evaluate(
                        current_training_iter=training_iteration, full_epoch_finished=False
                ):
                    self.__evaluation_and_saving(
                        dev_dataloader=dev_dataloader,
                        epoch=epoch,
                        iteration=training_iteration,
                    )
        return training_iteration

    def __train_step(self, batch_number: int, batch: Dict, epoch: int, iteration: int):
        self.model.train()

        model_step_output: ModelStepOutput = self.model_step(
            self.cfg, self.model, batch=batch, batch_number=batch_number
        )
        loss = model_step_output.loss
        prediction_scores = model_step_output.prediction_scores
        labels = model_step_output.gold_labels

        if self.is_multigpu_setting:
            loss = loss.mean()

        loss = loss / self.gradients_accumulation_steps
        self.scaler.scale(loss).backward()

        if (batch_number + 1) % self.gradients_accumulation_steps == 0:
            if self.cfg.clip_grad_norm:
                self.__apply_gradient_clipping()
            self.scaler.step(self.optimizer)
            scale = self.scaler.get_scale()
            self.scaler.update()
            skip_lr_scheduler_step = scale != self.scaler.get_scale()
            self.optimizer.zero_grad()
            if not skip_lr_scheduler_step:
                self.lr_scheduler.step()

        metric_values = (
            self.train_evaluation_metrics_accumulator.calculate_and_accumulate_metrics(
                gold_labels=labels, model_outcomes=prediction_scores, loss=loss.item()
            )
        )
        return metric_values

    def __apply_gradient_clipping(self):
        if self.cfg.clip_grad_norm and self.cfg.clip_grad_norm > 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.clip_grad_norm
            )

    def __evaluation_and_saving(self, dev_dataloader, epoch, iteration):
        dev_metrics = self.__evaluate(dev_dataloader, epoch, iteration)
        self.conditional_checkpoint_saver.check_and_store(
            model=self.model,
            metrics={m_name: m_value for m_name, m_value in dev_metrics.items()},
            epoch=epoch,
            iteration=iteration,
            metrics_for_checkpoint=self.metrics_holder.metrics_in_checkpoint_name,
        )

    def __evaluate(self, dataloader, epoch: int, iteration: int):
        logger.info(
            f"Evaluating the model after epoch {epoch}, iteration {iteration}..."
        )
        self.model.eval()
        with torch.no_grad():
            with batch_progress_bar(
                    epoch, self.cfg.num_epochs, dataloader, iteration
            ) as t:
                for batch_number, batch in enumerate(t):

                    model_step_output: ModelStepOutput = self.model_step(
                        self.cfg, self.model, batch=batch, batch_number=batch_number
                    )
                    loss = model_step_output.loss
                    model_outcomes = model_step_output.prediction_scores
                    labels = model_step_output.gold_labels
                    if self.is_multigpu_setting:
                        loss = loss.mean()

                    self.dev_evaluation_metrics_accumulator.calculate_and_accumulate_metrics(
                        labels, model_outcomes, loss.item()
                    )
            averaged_metrics = (
                self.dev_evaluation_metrics_accumulator.get_all_metrics_average(
                    reset_afterwards=True
                )
            )

        return averaged_metrics


def preinstantiation_stuff():
    logger.info(
        " >> WARNING: Filtering/Ignoring deprecation/future/undefined_metric warnings..."
    )
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    logger.info(' >> ATTENTION: Setting CUDA_DEVICE_ORDER="PCI_BUS_ID"')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def set_all_random_seeds_to(seed: int = 42):
    logger.info(f"Setting all random seeds (Python/Numpy/Pytorch) to: {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


class AutocastModel(Module):
    def __init__(self, module: Module, use_amp: bool):
        super(AutocastModel, self).__init__()
        self.module = module
        self.use_amp = use_amp

    def forward(self, *inputs, **kwargs):
        with autocast(enabled=self.use_amp):
            return self.module(*inputs, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class ExtendedDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def to_data_parallel(model: Module, use_amp: bool):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
    )
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        model = AutocastModel(model.cuda(), use_amp=use_amp)
        logger.info(f"Using all available cuda devices")
        device = torch.device(f"cuda:{0}")
        model = ExtendedDataParallel(model)

    model.to(device)

    return model


PROGRESS_BAR_NCOLS: int = 150


def batch_progress_bar(
        epoch: int, num_epochs: int, dataloader: DataLoader, global_iteration: int = None
):
    description = (
        f"Epoch {epoch}/{num_epochs} progress"
        if not global_iteration
        else f"Eval (epoch:{epoch}/{num_epochs}, train_iter:{global_iteration}) progress"
    )
    t = tqdm(
        dataloader,
        position=0,
        leave=True,
        desc=description,
        postfix="",
        ncols=PROGRESS_BAR_NCOLS,
    )
    return t


def report_to_progress_bar(
        progress_bar,
        metric_values,
        metrics_for_progress_bar: List[BaseEvaluationMetric],
        current_learning_rate: float,
):
    progress_bar_metrics_names: Set[str] = {
        metric.name for metric in metrics_for_progress_bar
    }
    postfix_str = "".join(
        [
            f"; {metric_name}:{metric_result:3.4f}"
            for metric_name, metric_result in metric_values
            if metric_name in progress_bar_metrics_names
        ]
    )
    progress_bar.postfix = postfix_str + f"; lr:{current_learning_rate:1.7f}"


def instantiate_adamw(model: Module, lr: float, correct_bias=False):
    optimizer_grouped_parameters = __get_parameters_for_the_optimizer(model)
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, correct_bias=correct_bias)
    return optimizer


def __get_parameters_for_the_optimizer(model):
    optimizer_grouped_parameters = __model_parameters_for_full_finetuning(model)
    return optimizer_grouped_parameters


def __model_parameters_for_full_finetuning(model):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay_rate": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay_rate": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def instantiate_linear_scheduler_with_warmup(
        optimizer: Optimizer,
        num_training_instances: int,
        config: MapaEntityDetectionTrainerConfig,
):
    warmup_steps = __calculate_warmup_steps(
        warmup_epochs=config.warmup_epochs,
        num_training_instances=num_training_instances,
        batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradients_accumulation_steps,
    )
    total_steps = __calculate_total_steps(
        total_epochs=config.num_epochs,
        num_training_instances=num_training_instances,
        batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradients_accumulation_steps,
    )
    lambda_lr = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        last_epoch=-1,
    )
    return lambda_lr


def instantiate_constant_lr_scheduler(optimizer: Optimizer):
    return get_constant_schedule(optimizer=optimizer, last_epoch=-1)


def __calculate_warmup_steps(
        warmup_epochs: float,
        num_training_instances: int,
        batch_size: int,
        gradient_accumulation_steps: int,
):
    real_batch_size = batch_size * gradient_accumulation_steps
    training_steps_per_epoch = num_training_instances / real_batch_size
    warmup_steps = int(warmup_epochs * training_steps_per_epoch)
    return warmup_steps


def __calculate_total_steps(
        total_epochs: int,
        num_training_instances: int,
        batch_size: int,
        gradient_accumulation_steps: int,
):
    real_batch_size = batch_size * gradient_accumulation_steps
    training_steps_per_epoch = num_training_instances / real_batch_size
    total_steps = int(total_epochs * training_steps_per_epoch)
    return total_steps


def main_train(MAPA_HIERARCHY):
    MAPA_ENTITIES_HIERARCHY = MapaEntitiesHierarchy(hierarchy_dict=MAPA_HIERARCHY)
    yaml_training_config: Configuration = confuse.Configuration(
        "MAPA single training", __name__
    )
    yaml_training_config.set_file(os.path.join("core", "training", "training_config.json"))
    training_config = MapaEntityDetectionTrainerConfig(**yaml_training_config.get(dict))

    trainer = MapaEntityDetectionTrainer(config=training_config, mapa_entites_hierarchy=MAPA_ENTITIES_HIERARCHY)
    trainer.train()
