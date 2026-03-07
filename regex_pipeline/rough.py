import pandas as pd
import re
from ftfy import fix_text

df = pd.read_csv("quotes_speakers_main.csv")

print(len(df))

