# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Provides utilities for SageMaker Pipeline CLI."""
from __future__ import absolute_import

import ast
import collections.abc
import copy
import logging
import os
import json
from typing import Dict, List

from jinja2 import Template

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def get_pipeline_driver(module_name, kwargs):
    """Gets the driver for generating your pipeline definition.

    Pipeline modules must define a get_pipeline() module-level method.

    Args:
        module_name: The module name of your pipeline.
        kwargs: Passed arguments that your pipeline templated by.

    Returns:
        The SageMaker Workflow pipeline.
    """
    _imports = __import__(module_name, fromlist=["get_pipeline"])
    return _imports.get_pipeline(**kwargs)


def convert_struct(str_struct=None):
    return ast.literal_eval(str_struct) if str_struct else {}


def _get_env_vars(prefix=''):
    env_vars = {}
    for key, value in os.environ.items():
        if prefix in key:
            env_vars[key.replace(prefix, '').lower()] = value
    return env_vars


def update_dict(original, updates) -> Dict:
    if original is None:
        return updates
    if updates is None:
        return original

    def inplace_update(o, u):
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping) and v:
                o[k] = inplace_update(o.get(k, {}), v)
            else:
                o[k] = v
        return o

    return inplace_update(copy.deepcopy(original), updates)


def read_json_config(*path_components) -> Dict:
    filename = os.path.join(*path_components)
    if not os.path.isfile(filename):
        logger.debug(f"Didn't find file {filename}")
        return {}
    if os.path.getsize(filename) == 0:
        logger.debug(f"Found empty config {filename}")
        return {}
    else:
        logger.debug(f"Reading config {filename}")
        with open(filename, 'r') as f:
            return json.loads(f.read())


def read_and_merge_config(base_path: str,
                          path_components: List[str]) -> Dict:
    base_template = read_json_config(base_path, "common_template.json")

    for i, component in enumerate(path_components):
        updates = read_json_config(base_path, *path_components[:i + 1], f"{component}_specific_updates.json")
        base_template = update_dict(base_template, updates)

    return base_template


def get_pipeline_args():
    env_vars = _get_env_vars()
    base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")
    merged_inherited_config = read_and_merge_config(base_path,
                                                    [env_vars.get('service'),
                                                     env_vars.get('pipeline_module_name'),
                                                     env_vars.get('model_name'),
                                                     env_vars.get('pipeline_type'),
                                                     env_vars.get('aws_region')])

    kwargs = {**env_vars, **merged_inherited_config}
    logger.info(f"Following configuration was loaded and merged\n" +
                json.dumps(kwargs, sort_keys=True, indent=2))

    template = Template(json.dumps(kwargs)).render(
        aws_region=env_vars.get('aws_region').replace('-', '_')
    )

    return json.loads(template)
