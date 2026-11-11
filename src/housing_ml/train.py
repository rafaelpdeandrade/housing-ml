import logging
from datetime import UTC, datetime

import pandas as pd
import sklearn
from sklearn.model_selection import kFold, cross_validate, train_test_split

from housing_ml import __version__
from housing_ml.config import Settings, get_settings
from housing_ml.data import load_dataset
from housing_ml.evaluate import evaluate
from housing_ml.features import FEATURES, TARGET
from housing_ml.logging_config import log_step, setup_logging
from housing_ml.model import build_model
from housing_ml.registry import save_artifact