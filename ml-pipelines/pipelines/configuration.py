from dataclasses import dataclass
from typing import List, Dict

INSTANCES_CONFIGURATION_KEY = "instances_configuration"
PROCESSING_INSTANCE_TYPE_KEY = "processing_instance_type"
PROCESSING_INSTANCE_COUNT_KEY = "processing_instance_count"
TRAINING_INSTANCE_TYPE_KEY = "training_instance_type"
TRAINING_INSTANCE_COUNT_KEY = "training_instance_count"
META_DATA_STEPS_INSTANCE_TYPE_KEY = "meta_data_steps_instance_type"
META_DATA_STEPS_INSTANCE_COUNT_KEY = "meta_data_steps_instance_count"
REGISTER_MODEL_INSTANCE_TYPES_KEY = "register_model_instance_types"


@dataclass
class MLInstancesConfiguration:
    processing_instance_type: str
    processing_instance_count: int
    training_instance_type: str
    training_instance_count: int
    meta_data_steps_instance_type: str
    meta_data_steps_instance_count: int
    register_model_instance_types: List[str]

    @classmethod
    def from_kwargs(cls,
                    kwargs_dict: Dict):
        instances_configuration = kwargs_dict[INSTANCES_CONFIGURATION_KEY]
        return cls(
            processing_instance_type=instances_configuration[PROCESSING_INSTANCE_TYPE_KEY],
            processing_instance_count=instances_configuration[PROCESSING_INSTANCE_COUNT_KEY],
            training_instance_type=instances_configuration[TRAINING_INSTANCE_TYPE_KEY],
            training_instance_count=instances_configuration[TRAINING_INSTANCE_COUNT_KEY],
            meta_data_steps_instance_type=instances_configuration[META_DATA_STEPS_INSTANCE_TYPE_KEY],
            meta_data_steps_instance_count=instances_configuration[META_DATA_STEPS_INSTANCE_COUNT_KEY],
            register_model_instance_types=instances_configuration[REGISTER_MODEL_INSTANCE_TYPES_KEY])
