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
### X, y로 데이터 나누기
```python
X_boston = boston.drop(columns="MEDV").values
y_boston = boston['MEDV'].to_frame().values
```
### train test set 나누기
```python
X_train, X_test, y_train, y_test = train_test_split(X_boston,
                                                    y_boston,
                                                    test_size=0.2,
                                                    random_state=0 # seed값 설정 -> 섞이는 순서를 동일
                                                   )
```
### 데이터 정규화
```python
#### Sklearn을 이용해 Standard Scaling 처리
scaler = StandardScaler()
scaler.fit(X_train)  # 어떻게 변환할지 학습 -> 평균/표준편차 계산
X_train_scaled = scaler.transform(X_train)  # 변환.
X_test_scaled = scaler.transform(X_test)
```
### torch.Tensor 타입으로 변경
```python
X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
```
### dataset 생성
```python
# Dataset 생성
boston_trainset = TensorDataset(X_train_scaled, y_train)
boston_testset = TensorDataset(X_test_scaled, y_test)
```
### dataloader 생성
```python
# DataLoader 생성
boston_trainloader = DataLoader(boston_trainset, 
                               batch_size=200,
                               shuffle=True,
                               drop_last=True)
boston_testloader = DataLoader(boston_testset, 
                               batch_size=len(boston_testset))
```

## 3. 모델정의
```python
class BostonModel(nn.Module):
    
    def __init__(self):
        # nn.Module의 __init__() 실행 => 초기화.
        super().__init__()
        # forward propagation(예측) 할때 필요한 Layer들 생성.
        
        ## 입력 feature: 13, 출력 feature: 32 => weight: 13(weight수) x 32(unit수)
        self.lr1 = nn.Linear(in_features=13, out_features=32)
        self.lr2 = nn.Linear(32, 16)
        ## lr3 -> 출력 Layer: out_features=모델이 출력해야할 값의 개수에 맞춰준다.
        self.lr3 = nn.Linear(16, 1) # 집값(중앙값) 하나를 예측해야 하므로 1로 설정.
        
    def forward(self, X):
        out = self.lr1(X)    # 선형
        out = nn.ReLU()(out) # 비선형
        out = self.lr2(out)  # 선형
        out = nn.ReLU()(out) # 비선형
        
        out = self.lr3(out) # 출력 레이어(이 값이 모델의 예측값이 된다.)
        # 회귀의 출력결과에는 Activation 함수를 적용하지 않는다.
        #    예외: 출력값의 범위가 정해져 있고 그 범위값을 출력하는 함수가 있을경우에는 적용 가능
        #       범위: 0 ~ 1 -> logistic (nn.Sigmoid())
        #            -1 ~ 1 -> tanh (nn.Tanh())
        return out
```
### 모델생성
```python
# 모델 생성
boston_model = BostonModel()
## 모델 구조 확인
print(boston_model)  # attribute로 설정된 Layer들을 확인.

BostonModel(
  (lr1): Linear(in_features=13, out_features=32, bias=True)
  (lr2): Linear(in_features=32, out_features=16, bias=True)
  (lr3): Linear(in_features=16, out_features=1, bias=True)
)
```
## 4. 학습(train)
```python
# 하이퍼파라미터 (우리가 설정하는 파라미터) 정의 
N_EPOCH = 1000
LR = 0.001

# 모델 준비
boston_model = boston_model.to(device)  # 모델: 1. 생성 2. device를 설정.
# loss 함수 정의 - 회귀: mse
loss_fn = nn.MSELoss()
# optimizer 정의
optimizer = torch.optim.RMSprop(boston_model.parameters(), lr=LR)
# torch.optim 모듈에 최적화알고리즘들이 정의. (모델의 파라미터, 학습률)

## 에폭별 학습 결과를 저장할 리스트 
## train loss와 validation loss 를 저장.
train_loss_list = []
valid_loss_list = []

import time
## Train (학습/훈련) 
### 두단계 -> Train + Validation => step별로 train -> epoch 별로 검증.
s = time.time()
for epoch in range(N_EPOCH):
    ### 한 epoch에 대한 train 코드
    ######################################
    # train - 모델을 train mode로 변경
    ######################################
    boston_model.train() # train 모드로 변경
    train_loss = 0.0 # 현재 epoch의 train loss를 저장할 변수
    ### batch 단위로 학습 => step
    for X, y in boston_trainloader:
        ## 한 STEP에 대한 train 코드
        # 1. X, y 를 device로 옮긴다. => 모델과 동일한 device에 위치시킨다.
        X, y = X.to(device), y.to(device)
        # 2. 모델 추정(예측) => forward propagation
        pred = boston_model(X)
        # 3. loss 계산
        loss = loss_fn(pred, y) # 오차계산 -> grad_fn
        # 4. 파라미터 초기화
        optimizer.zero_grad()
        # 5. back propagation -> 파라미터들의 gradient값들을 계산.
        loss.backward() # 모든 weight와 bias 에 대한 loss의 gradient들을 구한다. - 변수의 grad 속성에 저장.
        # 6. 파라미터 업데이트
        optimizer.step()
        
        # 7. 현 step의 loss를 train_loss에 누적
        train_loss += loss.item()
    # train_loss의 전체 평균을 계산 (step별 평균loss의 합계  -> step수로 나눠서 전체 평균으로 계산.)
    train_loss /= len(boston_trainloader) # step수로 나누기.
    
    ############################################
    # validation - 모델을 평가(eval) mode로 변경 
    #            - 검증, 평가, 서비스 할때.
    #            - validation/test dataset으로 모델을 평가.
    ############################################
    boston_model.eval() # 평가 모드로 변경.
    # 검증 loss를 저장할 변수
    valid_loss = 0.0
    # 검증은 gradient 계산할 필요가 없음. forward propagation시 도함수를 구할 필요가 없다.
    with torch.no_grad():
        for X_valid, y_valid in boston_testloader:
            # 1. X, y를 device로 이동.
            X_valid, y_valid = X_valid.to(device), y_valid.to(device)
            # 2. 모델을 이용해 예측
            pred_valid = boston_model(X_valid)
            # 3. 평가 - MSE
            valid_loss += loss_fn(pred_valid, y_valid).item()
        # valid_loss 평균
        valid_loss /= len(boston_testloader)
    # 현 epoch에 대한 학습 결과 로그를 출력 + list에 추가
    print(f"[{epoch+1}/{N_EPOCH}] train loss: {train_loss:.4f}, valid loss: {valid_loss:.4f}")
    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)

e = time.time()
```
## 5. 학습 시각화
```python
## train loss, valid loss 의 epoch 별 변화의 흐름 시각화.
plt.plot(range(1, N_EPOCH+1), train_loss_list, label="Train Loss")
plt.plot(range(1, N_EPOCH+1), valid_loss_list, label="Validation Loss")

plt.xlabel("EPOCH")
plt.ylabel("Loss")
# plt.ylim(3, 50)
plt.legend()
plt.show()
```
![image](https://github.com/jsj5605/regression/assets/141815934/86c4c37c-1f37-4c1b-8103-491cce4d3b24)

## 6. 모델 저장







