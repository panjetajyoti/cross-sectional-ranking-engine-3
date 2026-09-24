"""
Cross-Sectional Ranking & Propensity Engine
Package Initialization
"""

from .ingestion import fetch_market_data
from .features import build_feature_pipeline
from .labeling import run_labeling_pipeline
from .ranker_lambdamart import train_lambdamart_engine
from .calibration import run_calibration_pipeline
from .conformal import run_conformal_pipeline
from .ic_analytics import run_ic_analytics_pipeline
from .backtest import run_backtest_pipeline

__version__ = "1.0.0"