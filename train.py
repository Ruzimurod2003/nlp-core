import json
import os
from core.training_service import main_train


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "appsettings.json")
    with open(config_path, "r") as config_file:
        con = json.load(config_file)
    return con


MAPA_HIERARCHY = load_config().get("MAPA_HIERARCHY")
if __name__ == "__main__":
    main_train(MAPA_HIERARCHY)
