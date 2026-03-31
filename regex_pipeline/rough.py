import pandas as pd
import re
from ftfy import fix_text

df = pd.read_csv("clean_main_dataset_6.csv")

print(len(df))

