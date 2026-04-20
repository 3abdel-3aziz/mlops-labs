from fastapi import FastAPI
import pandas as pd
from src.models import ModelPredictor 
from src.features import NumericalHandeller, Encoder 