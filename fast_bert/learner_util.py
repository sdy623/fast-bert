import clearml
from clearml.utilities.plotly_reporter import SeriesInfo
import numpy as np
from typing import List, Optional

import torch
from pathlib import Path

import logging

from transformers import (
    AdamW,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)

from transformers.onnx.convert import export
from transformers.models.bert import BertConfig, BertOnnxConfig

from pytorch_lamb import Lamb


class Learner(object):
    def __init__(
        self,
        data,
        model,
        pretrained_model_path,
        output_dir,
        device,
        logger=logging.getLogger(__name__),
        multi_gpu=True,
        is_fp16=True,
        warmup_steps=0,
        fp16_opt_level="O1",
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        adam_epsilon=1e-8,
        logging_steps=100,
        clearML_task: Optional[clearml.Task] = None,
    ):

        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        self.data = data
        self.model = model
        self.pretrained_model_path = pretrained_model_path
        self.multi_gpu = multi_gpu
        self.is_fp16 = is_fp16
        self.fp16_opt_level = fp16_opt_level
        self.adam_epsilon = adam_epsilon
        self.warmup_steps = warmup_steps
        self.grad_accumulation_steps = grad_accumulation_steps
        self.device = device
        self.logger = logger
        self.layer_groups = None
        self.optimizer = None
        self.n_gpu = 0
        self.max_grad_norm = max_grad_norm
        self.logging_steps = logging_steps
        self.max_steps = -1
        self.weight_decay = 0.0
        self.model_type = data.model_type

        self.output_dir = output_dir

        if self.multi_gpu:
            self.n_gpu = torch.cuda.device_count()

    # Get the optimiser object
    def get_optimizer(self, lr, optimizer_type="lamb"):

        # Prepare optimiser and schedule
        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        if optimizer_type == "lamb":
            optimizer = Lamb(
                optimizer_grouped_parameters, weight_decay=0.1, lr=lr, eps=1e-12
            )
        elif optimizer_type == "adamw":
            optimizer = AdamW(
                optimizer_grouped_parameters, lr=lr, eps=self.adam_epsilon
            )

        return optimizer

    # Get learning rate scheduler
    def get_scheduler(self, optimizer, t_total, schedule_type="warmup_cosine"):

        SCHEDULES = {
            None: get_constant_schedule,
            "none": get_constant_schedule,
            "warmup_cosine": get_cosine_schedule_with_warmup,
            "warmup_constant": get_constant_schedule_with_warmup,
            "warmup_linear": get_linear_schedule_with_warmup,
            "warmup_cosine_hard_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
        }

        if schedule_type is None or schedule_type == "none":
            return SCHEDULES[schedule_type](optimizer)

        elif schedule_type == "warmup_constant":
            return SCHEDULES[schedule_type](
                optimizer, num_warmup_steps=self.warmup_steps
            )

        else:
            return SCHEDULES[schedule_type](
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=t_total,
            )

    def save_model(self, path=None):

        if not path:
            path = self.output_dir / "model_out"

        path.mkdir(exist_ok=True)

        # Convert path to str for save_pretrained calls
        path = str(path)

        torch.cuda.empty_cache()
        # Save a trained model
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Only save the model it-self
        model_to_save.save_pretrained(path)

        # save the tokenizer
        self.data.tokenizer.save_pretrained(path)

    def export_onnx(self, path=None):
        if not path:
            path = self.output_dir / "model_out"

        path.mkdir(exist_ok=True)

        torch.cuda.empty_cache()

        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )

        bert_onnx_config = BertOnnxConfig(model_to_save.config)
        # Handles all the above steps for you
        export(preprocessor=self.data.tokenizer, model=model_to_save,
               output=Path.joinpath(path, Path("model.onnx")), opset=15, device=self.device,
               config=bert_onnx_config)

    def upload_model(self, path=None, task: clearml.Task=None, model_name="bert-base-uncased", upload_uri=None):
        if not path:
            path = self.output_dir / "model_out"

        path.mkdir(exist_ok=True)

        torch.cuda.empty_cache()
        task_reporter_sc = task.get_all_reported_scalars()
        output_model = clearml.OutputModel(task=task, framework="PyTorch")

        acc_series = task_reporter_sc['accuracy_multilabel']['val']
        f1_series = task_reporter_sc['F1']['val']
        roc_auc_series = task_reporter_sc['ROC_AUC']['val']
        loss_series = task_reporter_sc['loss']['val']

        '''
        '''
        output_model.report_line_plot(
            title="Accuracy", series=[
                    SeriesInfo(
                        name="val",
                        data=np.column_stack(
                            (acc_series['x'], acc_series['y'])
                        )
            )],
            xaxis="Iterations",
            yaxis="Validation Accuracy",
        )

        output_model.report_line_plot(
            title="F1", series=[
                    SeriesInfo(
                        name="val",
                        data=np.column_stack(
                            (f1_series['x'], f1_series['y'])
                        )
            )],
            xaxis="Iterations",
            yaxis="Validation F1",
        )

        output_model.report_line_plot(
            title="AUC", series=[
                    SeriesInfo(
                        name="val",
                        data=np.column_stack(
                            (roc_auc_series['x'], roc_auc_series['y'])
                        )
            )],
            xaxis="Iterations",
            yaxis="Validation AUC",
        )

        output_model.report_line_plot(
            title="Loss", series=[
                    SeriesInfo(
                        name="val",
                        data=np.column_stack(
                            (loss_series['x'], loss_series['y'])
                        )
            )],
            xaxis="Iterations",
            yaxis="Validation Loss ",
        )

        output_model.report_single_value("Accuracy", acc_series['y'][-1])
        output_model.report_single_value("F1", f1_series['y'][-1])
        output_model.report_single_value("ROC_AUC", roc_auc_series['y'][-1])
        output_model.report_single_value("Loss", loss_series['y'][-1])
        #output_model.update_labels()
        output_model.update_weights(weights_filename=str(Path.joinpath(path, Path("model.safetensors"))), upload_uri=upload_uri)
        #task.update_output_model(model_path=str(Path.joinpath(path, Path("model.safetensors"))), model_name="model_name")
