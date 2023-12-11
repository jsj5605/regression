# regression
예측할 값이 정해져 있지 않는 경우. => 연속형값(실수)을 추론

## Boston Housing Dataset
보스턴 주택가격 dataset은 여러 속성을 바탕으로 해당 타운 주택 가격의 중앙값을 예측하는 문제.

## 1. import 
``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torchinfo

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
```

## 2. dataset, dataloader 생성
### dataset 불러오기 및 정보확인
```python
boston = pd.read_csv('data/boston_hosing.csv')
print(boston.shape)
boston.info()
boston.head()

	CRIM	ZN	INDUS	CHAS	NOX	RM	AGE	DIS	RAD	TAX	PTRATIO	B	LSTAT	MEDV
0	0.00632	18.0	2.31	0.0	0.538	6.575	65.2	4.0900	1.0	296.0	15.3	396.90	4.98	24.0
1	0.02731	0.0	7.07	0.0	0.469	6.421	78.9	4.9671	2.0	242.0	17.8	396.90	9.14	21.6
2	0.02729	0.0	7.07	0.0	0.469	7.185	61.1	4.9671	2.0	242.0	17.8	392.83	4.03	34.7
3	0.03237	0.0	2.18	0.0	0.458	6.998	45.8	6.0622	3.0	222.0	18.7	394.63	2.94	33.4
4	0.06905	0.0	2.18	0.0	0.458	7.147	54.2	6.0622	3.0	222.0	18.7	396.90	5.33	36.2
```


