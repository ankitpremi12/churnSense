import sys
import traceback
import pandas as pd

from data.load import load_raw

try:
    df = load_raw()
    print("shape:", df.shape)
    with pd.option_context('display.max_columns', 30):
        print(df.head().to_string())
except Exception as e:
    print('ERROR:', e)
    traceback.print_exc()
