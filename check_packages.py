def install_packages():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '.'])

try:
    import subprocess
    import fastapi
    import uvicorn
    import pydantic
    import json
    import logging
    import os
    import logging
    import re
    import itertools
    import time
    import warnings
    import torch
    import os.path
    import confuse
    import sys
    import random
    import shutil
    import numpy as np
except ImportError:
    install_packages()