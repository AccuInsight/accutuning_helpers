# accutuning_helpers
accutuning preprocessors, utils for public

### Prerequisite
```
$ apt-get install -y libgomp1
```

### Install
```
$ pip3 install https://github.com/AIIP-DEV/accutuning_helpers/archive/<version>.zip
```

### Example
```
import pickle 
import pandas as pd

# x_test, y_test : 예측용 데이터셋
x_test = pd.read('x_test.csv')
y_test = pd.read('y_test.csv')

# pipeline.pkl : Acctutuning에서 배포된 모델 
pipeline = pickle.load(open('pipeline.pkl', 'rb'))

pred = pipeline.predict(x_test)
print(f'Accuracy: {(pred==y_test).mean()}')
```
