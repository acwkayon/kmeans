import os
import pandas as pd
from io import StringIO

ret = os.popen("python train.py 3 iris.csv")
df = pd.read_csv(StringIO(ret.read()))


def test_train_model_rows():
    assert df.shape[0] == 3


def test_train_model_columns():
    assert df.shape[1] == 5


def test_train_model_labels():
    assert df['label'].values.tolist() == [0, 1, 2]


def test_train_model_column_names():
    assert list(df) == ['sepal_length', 'sepal_width',
                        'petal_length', 'petal_width',
                        'label']
