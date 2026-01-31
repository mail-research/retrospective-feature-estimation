from datetime import datetime
from functools import partial
from typing import Any

import attrs
import clearml as cml
import einops as ei
import numpy as np

from .training import TaskResult


@attrs.define
class Logger:
    run_name: str
    method_name: str
    dataset_name: str
    method_args: dict[str, Any]
    dataset_args: dict[str, Any]

    tkey_to_result: dict[str, TaskResult] = attrs.field(init=False, factory=dict)

    @property
    def unique_run_name(self):
        return f'{self.run_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    def log(self, tkey: str, task_result: TaskResult):
        self.tkey_to_result[tkey] = task_result

    def export(self):
        return {
            'method_name': self.method_name,
            'method_args': self.method_args,
            'dataset_name': self.dataset_name,
            'dataset_args': self.dataset_args,
            'task_logs': {
                tkey: attrs.asdict(task_result) for tkey, task_result in self.tkey_to_result.items()
            },
        }

    def finish(self):
        pass


@attrs.define
class ClearMLLogger(Logger):
    task: cml.Task = attrs.field(init=False)

    @task.default  # ty:ignore[call-non-callable]
    def _init_task(self):
        task: cml.Task = cml.Task.init(
            project_name=f'test_cl/{self.method_name}/{self.dataset_name}',
            task_name=self.unique_run_name,
            tags=[f'{k}:{v}' for k, v in self.method_args.items()],
            auto_connect_arg_parser=False,
            auto_resource_monitoring=False,
            auto_connect_streams={'stdout': True, 'stderr': False, 'logging': True},
        )
        task.connect(self.method_args | self.dataset_args)
        task.set_script(diff='')  # remove git diff
        return task

    @property
    def logger(self):
        return self.task.get_logger()

    def log(self, tkey: str, task_result: TaskResult):
        super().log(tkey, task_result)
        for name, values in task_result.training.items():
            self.logger.report_scatter2d(
                title=f'training/{tkey}',
                series=name,
                scatter=ei.pack([np.arange(1, len(values) + 1), np.asarray(values)], 'd *')[0],
                iteration=None,
                xaxis='iteration',
                yaxis='value',
                mode='lines',
            )

        _report_historgram = partial(
            self.logger.report_histogram,
            series=tkey,
            xaxis='after each task',
            yaxis='value',
            iteration=None,
        )
        for name, values in task_result.validation_agg.items():
            _report_historgram(
                title=f'validation/{name}',
                values=np.asarray(values),
                xlabels=task_result.validation.keys(),
            )

        for name, values in task_result.test_agg.items():
            _report_historgram(
                title=f'test/{name}',
                values=np.asarray(values),
                xlabels=task_result.test.keys(),
            )

        for name, value in task_result.validation_summary.items():
            self.logger.report_scalar(
                title='summary/validation',
                series=name,
                value=value,
                iteration=len(task_result.validation),
            )

        for name, value in task_result.test_summary.items():
            self.logger.report_scalar(
                title='summary/test',
                series=name,
                value=value,
                iteration=len(task_result.test),
            )

        self.task.upload_artifact(name=self.run_name, artifact_object=self.export())

    def finish(self):
        return self.task.close()
