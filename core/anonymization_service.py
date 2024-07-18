import json
import logging
import os
import re
import itertools
import time
import warnings
import torch
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from math import ceil
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple, Type, Union
from torch.utils.data import Dataset
from sklearn.exceptions import UndefinedMetricWarning
from torch import Tensor, nn
from torch.cuda.amp import autocast
from torch.nn import CrossEntropyLoss, Module
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BertConfig,
    BertModel,
    BertPreTrainedModel,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from dataclasses_serialization.json import JSONSerializer
from core.ProcessDocumentViewModel import ProcessDocumentResponseViewModel


def init_logger():
    FORMAT = "%(asctime)s %(levelname)s (%(threadName)s) %(module)s: %(message)s"
    logging.basicConfig(format=FORMAT, level=logging.INFO)


init_logger()
logger = logging.getLogger(__name__)

Tokens = List[str]
Labels = List[str]

CharSpan = Tuple[int, int]
TokenList = List[str]

EXTRA_RESOURCE_APPENDIX = "_extra_resource"


@dataclass
class MapaSeqLabellingOutput:
    tokens: List[str]
    level1_tags: List[str]
    level2_tags: List[str]


@dataclass
class EntityDetectionResult:
    text: str
    tokens: List[str]
    spans: List[Tuple[int, int]]
    level1_tags: List[str]
    level2_tags: List[str]


@dataclass
class MapaEntity:
    id: int
    entity_text: str
    entity_type: str
    offset: int


@dataclass
class MapaResult:
    text: str
    level1_entities: List[MapaEntity]
    level2_entities: List[MapaEntity]
    num_tokens: Optional[int] = None
    using_gpu: Optional[bool] = None


def main_service(text: str):
    start_time = time.time()
    mapa_result: MapaResult = anonymize(text=text)
    end_time = time.time()

    annotation_time = end_time - start_time
    num_whitespace_tokens = len(text.split())
    words_per_second = num_whitespace_tokens / annotation_time

    return ProcessDocumentResponseViewModel(
        text=mapa_result.text,
        elapsedSeconds=str(round(annotation_time, 3)),
        wordsPerSecond=str(round(words_per_second, 3)),
        usedDevice="GPU" if mapa_result.using_gpu else "CPU",
        level1Entities=mapa_result.level1_entities,
        level2Entities=mapa_result.level2_entities,
    )


def obtain_entities_from_bio_tagging(
        text: str, spans: List[CharSpan], labels: List[str]
):
    mapa_entities: List[MapaEntity] = []
    current_type = ""
    current_span_start = -1
    current_span_end = -1
    for i, label in enumerate(labels):
        start, end = spans[i]
        if label.startswith("B-") or label == "O":
            if current_type != "":
                mapa_entities.append(
                    MapaEntity(
                        id=len(mapa_entities),
                        entity_text=text[current_span_start:current_span_end],
                        offset=current_span_start,
                        entity_type=current_type,
                    )
                )
                current_type = ""
                current_span_start = -1
                current_span_end = -1
            if label.startswith("B-"):
                current_type = label.replace("B-", "")
                current_span_start = start
                current_span_end = end
        elif label.startswith("I-"):
            current_span_end = end

    if current_type != "":
        mapa_entities.append(
            MapaEntity(
                id=len(mapa_entities),
                entity_text=text[current_span_start:current_span_end],
                offset=current_span_start,
                entity_type=current_type,
            )
        )

    return mapa_entities


def tokenize(text: str):
    trans = re.sub(re.compile(r"(\s)"), r"_SEPARATOR_\1_SEPARATOR_", text)
    trans = re.sub(re.compile(r"([^\w])"), r"_SEPARATOR_\1_SEPARATOR_", trans)
    trans = re.sub(re.compile(r"(?:_SEPARATOR_)+"), "_SEPARATOR_", trans)
    tokens: List[str] = trans.split("_SEPARATOR_")
    tokens = [token for token in tokens if token != ""]
    tokens = [token for token in tokens if token != " "]
    return tokens


def calculate_offsets(raw_text, tokens):
    token_spans: List[Tuple[int, int]] = []
    latest_position = 0
    for token in tokens:
        if token == "[LINE_BREAK]":
            token_spans.append((-1, -1))
        token = token.strip()
        for match in re.finditer(re.escape(token), raw_text[latest_position:]):
            span_start = latest_position + match.start()
            span_end = span_start + len(token)
            token_spans.append((span_start, span_end))
            latest_position = token_spans[-1][0] + len(token)
            break
    for i, (start, end) in enumerate(token_spans):
        if start == -1 and end == -1:
            new_start = token_spans[i - 1][1] if i > 0 else 0
            new_end = new_start + 1
            token_spans[i] = (new_start, new_end)
    return token_spans


def do_pretokenization(raw_text: str):
    tokens: List[str] = tokenize(raw_text)
    token_spans = calculate_offsets(raw_text, tokens)
    tokens_with_spans: List[Tuple[str, CharSpan]] = list(zip(tokens, token_spans))
    return tokens_with_spans


class EnhancedTwoFlatLevelsSequenceLabellingConfig(BertConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_level2_labels = None
        self.num_level1_labels = None


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
        )  # + outputs[2:]
        return ((loss,) + output) if loss is not None else output

    def _reorder_cache(self, past, beam_idx):
        pass


class MAPATwoFlatLevelsInferencer:
    def __init__(
            self,
            model_path: str,
            valid_seq_len: int,
            ctx_window_len: int,
            batch_size: int,
            cuda_device_num=-1,
            model_class=EnhancedTwoFlatLevelsSequenceLabellingModel,
            mask_level2: bool = False,
            use_amp: bool = True,
            force_level1_from_level2: bool = False,
    ):
        preinstantiation_stuff()
        model_loader = ModelLoader(model_class)
        model_and_resources: ModelAndResources = model_loader.load_checkpoint(
            model_path, load_tokenizer=True
        )
        self.model = model_and_resources.model
        self.tokens_field = model_and_resources.data_fields["tokens"]
        self.level1_labels_field = model_and_resources.data_fields["level1_tags"]
        self.level2_labels_field = model_and_resources.data_fields["level2_tags"]

        self.mapa_entities = MapaEntitiesHierarchy.from_json(
            model_and_resources.extra_resources["mapa_entities"]
        )

        self.cuda_device_num = cuda_device_num
        self.device = (
            f"cuda:{cuda_device_num}"
            if torch.cuda.is_available() and cuda_device_num >= 0
            else "cpu"
        )

        self.model.to(self.device)
        self.model = AutocastModel(
            self.model, use_amp=use_amp and "cuda" in self.device
        )

        self.tokenizer = model_and_resources.tokenizer
        self.valid_seq_len = valid_seq_len
        self.ctx_window_len = ctx_window_len
        self.batch_size = batch_size

        self.mask_level2 = mask_level2
        if self.mask_level2:
            self.precomputed_level2_masks = self.precompute_level2_masks()
        self.force_level1_from_level2 = force_level1_from_level2

    def precompute_level2_masks(self):
        level2_out_idx = self.level2_labels_field.stoi(
            self.level2_labels_field.default_value
        )
        default_tensor = torch.tensor([level2_out_idx], dtype=torch.long)

        level2_masks = {
            level1_tag_idx: torch.zeros([len(self.level2_labels_field.vocab)]).scatter(
                dim=0, index=default_tensor, value=1
            )
            for level1_tag_idx in self.level1_labels_field.reverse_vocab.keys()
        }
        for L1, L2_values in self.mapa_entities.to_bio_dict().items():
            L2_values.add(self.level2_labels_field.default_value)
            if len(L2_values) != 0:
                level1_idx: int = self.level1_labels_field.stoi(L1)
                level2_values_idx: List[int] = [
                    self.level2_labels_field.stoi(val) for val in L2_values
                ]
                mask = torch.zeros([len(self.level2_labels_field.vocab)]).scatter(
                    dim=0,
                    index=torch.tensor(level2_values_idx, dtype=torch.long),
                    value=1,
                )
                level2_masks[level1_idx] = mask
        level2_masks = {key: val.to(self.device) for key, val in level2_masks.items()}
        level2_masks[
            self.level1_labels_field.stoi(self.level1_labels_field.default_value)
        ] = (
            torch.zeros([len(self.level2_labels_field.vocab)])
            .scatter(dim=0, index=default_tensor, value=1)
            .to(self.device)
        )

        return level2_masks

    @torch.no_grad()
    def make_inference(
            self, pretokenized_input: List[List[str]]
    ) -> List[MapaSeqLabellingOutput]:
        positions_of_empty_seqs = [
            seq_num for seq_num, seq in enumerate(pretokenized_input) if len(seq) == 0
        ]
        pretokenized_input = [seq for seq in pretokenized_input if len(seq) > 0]

        dataset = TwoFlatLevelsDataset.load_for_inference(
            pretokenized_input=pretokenized_input,
            tokenizer=self.tokenizer,
            valid_seq_len=self.valid_seq_len,
            ctx_len=self.ctx_window_len,
            input_field=self.tokens_field,
            level1_tags_field=self.level1_labels_field,
            level2_tags_field=self.level2_labels_field,
        )
        dataloader = DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        all_seq_nums: List[int] = []
        all_token_inputs: List[Tensor] = []
        all_level1_labels: List[Tensor] = []
        all_level2_labels: List[Tensor] = []

        for batch_num, batch in enumerate(dataloader):
            start_time = time.time()
            print(f"Processing batch {batch_num} of {len(dataloader)}...")
            instance_nums: Tensor = batch[WindowedSequencesGenerator.INSTANCE_IDX]
            input_ids = batch[self.tokens_field.name].to(device=self.device)
            ctx_mask = batch[self.tokens_field.name + "_ctx_mask"].to(
                device=self.device
            )
            input_mask = batch[self.tokens_field.attn_mask_name()].to(
                device=self.device
            )

            outputs = self.model(
                input_ids=input_ids, attention_mask=input_mask, ctx_mask=ctx_mask
            )
            level1_tags_logits, level2_tags_logits = outputs[:2]

            level1_tags_probs, predicted_level1_tags_labels = torch.max(
                level1_tags_logits.softmax(dim=2), dim=2
            )
            if self.mask_level2:
                level2_masks = torch.stack(
                    dim=0,
                    tensors=[
                        torch.stack(
                            dim=0,
                            tensors=[
                                self.precomputed_level2_masks[L1.item()] for L1 in seq
                            ],
                        )
                        for seq in predicted_level1_tags_labels
                    ],
                )

                masked_level2_tags_prob_dist = (
                        level2_tags_logits * level2_masks
                ).softmax(dim=2)
                level2_tags_probs, predicted_level2_tags_labels = torch.max(
                    masked_level2_tags_prob_dist, dim=2
                )
            else:
                level2_tags_probs, predicted_level2_tags_labels = torch.max(
                    level2_tags_logits.softmax(dim=2), dim=2
                )

            all_token_inputs.append(input_ids)
            all_level1_labels.append(predicted_level1_tags_labels)
            all_level2_labels.append(predicted_level2_tags_labels)

            all_seq_nums += instance_nums.tolist()

            end_time = time.time()
            print(f"Batch time: {end_time - start_time:.4f} seconds")

        rebuilt_level1_label_sequences = (
            WindowedSequencesGenerator.rebuild_original_sequences(
                seq_indices=all_seq_nums,
                windowed_sequences=torch.cat(all_level1_labels, dim=0),
                left_ctx_len=self.ctx_window_len + 1,
                right_ctx_len=self.ctx_window_len + 1,
                pad_idx=None,
            )
        )

        rebuilt_level2_label_sequences = (
            WindowedSequencesGenerator.rebuild_original_sequences(
                seq_indices=all_seq_nums,
                windowed_sequences=torch.cat(all_level2_labels, dim=0),
                left_ctx_len=self.ctx_window_len + 1,
                right_ctx_len=self.ctx_window_len + 1,
                pad_idx=None,
            )
        )

        level1_label_instances = [
            {
                self.level1_labels_field.name: [
                    self.level1_labels_field.itos(label) for label in labels
                ]
            }
            for labels in rebuilt_level1_label_sequences
        ]
        level2_label_instances = [
            {
                self.level2_labels_field.name: [
                    self.level2_labels_field.itos(label) for label in labels
                ]
            }
            for labels in rebuilt_level2_label_sequences
        ]
        merged_instances = [
            {**level1_label_instances[i], **level2_label_instances[i]}
            for i in range(len(level1_label_instances))
        ]

        remapped_instances: List[Dict[str, List]] = dataset.tokens_remapper.remap(
            instances_to_remap=merged_instances
        )

        decoded_instances = [
            {
                self.tokens_field.name: instance[self.tokens_field.name],
                self.level1_labels_field.name: instance[self.level1_labels_field.name],
                self.level2_labels_field.name: instance[self.level2_labels_field.name],
            }
            for instance in remapped_instances
        ]

        for empty_seq_position in positions_of_empty_seqs:
            decoded_instances.insert(
                empty_seq_position,
                {
                    self.tokens_field.name: [],
                    self.level1_labels_field.name: [],
                    self.level2_labels_field.name: [],
                },
            )

        self.__postprocess_line_breaks_and_x_(decoded_instances)
        [
            self.__fix_bio_tagging_issues_(instance[self.level1_labels_field.name])
            for instance in decoded_instances
        ]
        [
            self.__fix_bio_tagging_issues_(instance[self.level2_labels_field.name])
            for instance in decoded_instances
        ]
        if self.force_level1_from_level2:
            self.__force_level1_from_level2_(decoded_instances)
        result: List[MapaSeqLabellingOutput] = [
            MapaSeqLabellingOutput(
                tokens=inst[self.tokens_field.name],
                level1_tags=inst[self.level1_labels_field.name],
                level2_tags=inst[self.level2_labels_field.name],
            )
            for inst in decoded_instances
        ]
        return result

    def __postprocess_line_breaks_and_x_(self, instances: List[Dict[str, List[str]]]):
        for instance in instances:
            tokens = instance[self.tokens_field.name]
            level1_tags = instance[self.level1_labels_field.name]
            level2_tags = instance[self.level2_labels_field.name]
            for i, token in enumerate(tokens):
                if token == "\n":
                    tokens[i] = "[LINE_BREAK]"
                    level1_tags[i] = "O"
                    level2_tags[i] = "O"
                if level1_tags[i] == "X":
                    level1_tags[i] = "O"
                    level2_tags[i] = "O"

    @classmethod
    def __fix_bio_tagging_issues_(cls, labels: List[str]):
        for i, bio_label in enumerate(labels):
            if "-" in bio_label:
                bio, label = bio_label.split("-", maxsplit=1)
                if i == 0 or labels[i - 1] == "O":
                    bio = "B"
                labels[i] = f"{bio}-{label}"

    def __force_level1_from_level2_(self, instances: List[Dict[str, List[str]]]):
        for instance in instances:
            level1_tags = instance[self.level1_labels_field.name]
            level2_tags = instance[self.level2_labels_field.name]
            for i, level1_tag in enumerate(level1_tags):
                level2_tag = level2_tags[i]
                if level1_tag == "O" and level2_tag != "O":
                    level1_wo_bio = self.mapa_entities.get_level2_parent(
                        level2_tag, lowercase=True, input_is_bio_tagged=True
                    )
                    level1_tags[
                        i
                    ] = f'{level2_tag.split("-", maxsplit=1)[0]}-{level1_wo_bio}'  # assuming that every tag that is not O has '-'


def detect_entities(
        text,
        tokens: List[str],
        spans: List[Tuple[int, int]],
        nerc_inferencer: MAPATwoFlatLevelsInferencer,
):
    nerc_output: MapaSeqLabellingOutput = nerc_inferencer.make_inference([tokens])[0]
    level1_tags, level2_tags = nerc_output.level1_tags, nerc_output.level2_tags

    entity_detection_result = EntityDetectionResult(
        text=text,
        tokens=tokens,
        spans=spans,
        level1_tags=level1_tags,
        level2_tags=level2_tags,
    )
    return entity_detection_result


def anonymize(text: str):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    tokens, spans = zip(*do_pretokenization(raw_text=text.strip()))

    nerc_inferencer = MAPATwoFlatLevelsInferencer(
        model_path=os.path.join("core", "model"),
        valid_seq_len=300,
        ctx_window_len=100,
        batch_size=4,
        cuda_device_num=0,
        model_class=EnhancedTwoFlatLevelsSequenceLabellingModel,
        mask_level2=True,
        use_amp=True,
        force_level1_from_level2=True,
    )

    tokens, spans = zip(
        *[(token, spans[i]) for i, token in enumerate(tokens) if token != "\n"]
    )
    detection_result: EntityDetectionResult = detect_entities(
        text=text, tokens=tokens, spans=spans, nerc_inferencer=nerc_inferencer
    )
    level1_tags, level2_tags = (
        detection_result.level1_tags,
        detection_result.level2_tags,
    )

    level1_entities: List[MapaEntity] = obtain_entities_from_bio_tagging(
        text, spans, level1_tags
    )
    level2_entities: List[MapaEntity] = obtain_entities_from_bio_tagging(
        text, spans, level2_tags
    )

    using_gpu = "cuda" in nerc_inferencer.device
    mapa_result = MapaResult(
        text=text,
        level1_entities=level1_entities,
        level2_entities=level2_entities,
        num_tokens=len(tokens),
        using_gpu=using_gpu,
    )
    return mapa_result


class MapaEntitiesHierarchy:
    def __init__(
            self,
            hierarchy_dict: Dict[str, Set[str]],
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

    def get_level2_parent(
            self, level2_label, lowercase: bool, input_is_bio_tagged: bool = False
    ):
        if input_is_bio_tagged:
            level2_label = level2_label.replace("B-", "").replace("I-", "")
        level1_label: str = self.__reversed_hierarchy[level2_label]
        if lowercase:
            level1_label = level1_label.lower()
        return level1_label

    def to_bio_dict(self):
        bio_dict = {}
        for level1, level2_values in self.__hierarchy_dict.items():
            bio_level2_values = set(
                [f"B-{value}" for value in level2_values]
                + [f"I-{value}" for value in level2_values]
            )
            bio_dict[f"B-{level1}"] = bio_level2_values
            bio_dict[f"I-{level1}"] = bio_level2_values
        return bio_dict

    @classmethod
    def from_json(cls, json_dict: Dict):
        return MapaEntitiesHierarchy(json_dict)


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

    def attn_mask_name(self):
        return "{}_attn_mask".format(self.name)

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
    ) -> Dict[str, int]:
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
        # instantiate the vocabulary
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
            return list(
                label_counter.keys()
            )  # if no min_support provided, return keys as-is, unfiltered
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

    def itos(self, index: int):
        if index in self.reverse_vocab:
            return self.reverse_vocab[index]
        else:
            raise Exception(
                f"Non-existent index value:{index} requested for a vocabulary."
            )

    @classmethod
    def deserialize_from_file(cls, base_path, field_name: str):
        path = os.path.join(base_path, f"{field_name}{cls.STORE_NAME_APPENDIX}.json")
        with open(path, "r", encoding="utf-8") as f:
            deserialized_dict = json.loads(f.read())
        return JSONSerializer.deserialize(cls, deserialized_dict)


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
        absolute_max_len = 512 - 2  # minus two to account for the CLS and SEP tokens
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
    def load_for_inference(
            cls,
            pretokenized_input: List[List[str]],
            tokenizer: PreTrainedTokenizer,
            valid_seq_len: int,
            ctx_len: int,
            input_field: FieldDefinition,
            level1_tags_field: FieldDefinition,
            level2_tags_field: FieldDefinition,
    ):
        instances = [
            {
                input_field.name: tokens,
                level1_tags_field.name: ["O"] * len(tokens),
                level2_tags_field.name: ["O"] * len(tokens),
            }
            for tokens in pretokenized_input
        ]
        return TwoFlatLevelsDataset(
            instances,
            tokenizer,
            valid_seq_len,
            ctx_len,
            input_field,
            level1_tags_field,
            level2_tags_field,
            build_vocabs=False,
            entities_hierarchy=None,
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


class TokenPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output


def preinstantiation_stuff():
    logger.info(
        " >> WARNING: Filtering/Ignoring deprecation/future/undefined_metric warnings..."
    )
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    logger.info(' >> ATTENTION: Setting CUDA_DEVICE_ORDER="PCI_BUS_ID"')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


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

    def load_checkpoint(self, model_path, load_tokenizer=False) -> ModelAndResources:
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

    def __load_data_fields(self, model_path) -> Dict[str, FieldDefinition]:
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

    def __load_extra_resources(self, model_path) -> Dict[str, Any]:
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

    def remap(
            self,
            instances_to_remap: List[Dict[str, List[str]]],
            fields_to_ignore: List[str] = None,
    ) -> List[Dict[str, List[str]]]:
        remapped_instances = []
        for i, instance in enumerate(instances_to_remap):
            try:
                remapped_instance = {self.tokens_field_name: self.original_tokens[i]}
                for field in list(self.tag_fields_names):
                    if fields_to_ignore and field in fields_to_ignore:
                        continue
                    sequence = instance[field]
                    remapped_instance[field] = [
                        sequence[idx] for idx in self.orig_to_tok_mappings[i]
                    ]
                remapped_instances.append(remapped_instance)
            except IndexError as e:
                logger.error(
                    "Some instance has given remapping problems, returning an empty instance",
                    exc_info=e,
                )
                remapped_instances.append(
                    {
                        self.tokens_field_name: [],
                        **{
                            tag_field_name: []
                            for tag_field_name in self.tag_fields_names
                        },
                    }
                )
        return remapped_instances


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
    ) -> List[Dict[str, List[str]]]:
        converted_instances: List[Dict[str, List[str]]] = []
        logger.info(f"Converting instances to windowed sequences with context...")
        for instance_num, instance in enumerate(tqdm(instances)):
            converted_instances += self.__transform_instance(instance_num, instance)
        return converted_instances

    def __transform_instance(
            self, instance_num, instance: Dict[str, List[str]]
    ) -> List[Dict[str, List[str]]]:
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
    ) -> List[List[str]]:
        windows: List[List[str]] = []
        for i in range(0, len(original_sequence), self.valid_seq_len):
            windows.append(original_sequence[i: i + self.valid_seq_len])
        windows_with_ctx: List[List[str]] = self.__add_ctx_windows(
            windows=windows, field=field
        )
        return windows_with_ctx

    def __add_ctx_windows(
            self, windows: List[List[str]], field: FieldDefinition
    ) -> List[List[str]]:
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
    ) -> List[str]:
        n = min(ceil(self.ctx_len / self.valid_seq_len), window_num)
        prev_ctx: List[str] = list(
            itertools.chain.from_iterable(windows[window_num - n: window_num])
        )[-self.ctx_len:]
        prev_ctx = [field.pad_token] * (self.ctx_len - len(prev_ctx)) + prev_ctx
        assert len(prev_ctx) == self.ctx_len
        return prev_ctx

    def __obtain_post_ctx(
            self, window_num, windows, field: FieldDefinition
    ) -> List[str]:
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
        ww_ctx = window_with_ctx  # just a rename
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
    def rebuild_original_sequences(
            cls,
            seq_indices: List[int],
            windowed_sequences: Tensor,
            left_ctx_len: int,
            right_ctx_len: int,
            pad_idx: Optional[int],
    ) -> List[List]:
        seq_split_points = cls.__calculate_seq_split_points(seq_indices)
        rebuilt_sequences: List[List] = []
        current_first = 0
        total_ctx_len = left_ctx_len + right_ctx_len
        for split_point in seq_split_points:
            current_seq_windows_with_ctx = windowed_sequences[
                                           current_first:split_point, :
                                           ]
            current_seq_windows = (
                torch.narrow(
                    current_seq_windows_with_ctx,
                    dim=1,
                    start=left_ctx_len,
                    length=current_seq_windows_with_ctx.shape[1] - total_ctx_len,
                )
                .contiguous()
                .view(-1)
            )
            if pad_idx:
                non_padding_mask = current_seq_windows.ne(pad_idx)
                current_seq_windows = torch.masked_select(
                    current_seq_windows, mask=non_padding_mask
                )
            rebuilt_sequences.append(current_seq_windows.flatten().tolist())
            current_first = split_point
        return rebuilt_sequences

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
