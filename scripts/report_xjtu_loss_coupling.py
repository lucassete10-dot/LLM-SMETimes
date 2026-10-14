"""Report matched loss-gradient experiments, with checkpoint replay checks."""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from battery_smetimes.data.cycle_time import CycleTimeWindowDataset
from battery_smetimes.models.loss_coupling import LossSeparatedHealthModel
from battery_v2_features import load_cell, load_manifest
from scripts.report_xjtu_cycle_time import aligned
from scripts.train_mit_joint_trajectory import save_json
from scripts.train_xjtu_cycle_time import HORIZONS, LOOKBACK, OUT as PREVIOUS
