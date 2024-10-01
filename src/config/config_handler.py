import yaml
import os
from typing import Dict


def load_config() -> Dict[str, any]:
    """
    Constructs the path to the configurations YAML file, reads the file, and returns its contents as a dictionary.

    Returns:
    - Dict[str, any]: The configuration settings loaded from the YAML file.
    """
    config_path = os.path.join(os.getcwd(), "src/config/config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config
