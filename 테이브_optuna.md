[OPTUNA]
========

#### optuna 프레임워크를 활용해 하이퍼 파라미터 최적화를 진행한다.

### 1. optuna 프레임워크 설치 및 불러오기

```python
!pip install optuna
import optuna
```

### 2. objective 함수 정의

##### 함수 내부 params 안에 파라미터를 정의하고, 랜덤한 파라미터 값으로 모델을 학습하고, validation set을 통해 구해진 f1_score이 반환되는 함수이다.
##### 파라미터 정의 시에는 해당 파라미터의 형태와 범위를 같이 설정한다.

```python
# create trial function
OPTUNA_OPTIMIZATION = True

def objective(trial):
    train_x, valid_x, train_y, valid_y = train_test_split(x_train,y_train, test_size=0.3)
    
    #define parameters
    params = {
        'iterations':trial.suggest_int("iterations", 500, 3000),
        'objective':trial.suggest_categorical('objective',['CrossEntropy','Logloss']),
        'bootstrap_type':trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
        'od_wait':trial.suggest_int('od_wait', 500, 1000),
        'learning_rate' : trial.suggest_uniform('learning_rate',0.01,1),
        'reg_lambda': trial.suggest_uniform('reg_lambda',1e-5,100),
        'random_strength': trial.suggest_uniform('random_strength',20,50),
        'depth': trial.suggest_int('depth',1,15),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',1,20),
        'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations',1,15),
        'verbose': False,
        "eval_metric":'F1',
        "cat_features" : cat_features,
        "one_hot_max_size":trial.suggest_int("one_hot_max_size",1,5),
        'task_type' : 'GPU',
    }
    
    if params['bootstrap_type'] == 'Bayesian':
        params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 10)
    elif params['bootstrap_type'] == 'Bernoulli':
        params['subsample'] = trial.suggest_float('subsample', 0.1, 1)
    
    # model fit
    model = CatBoostClassifier(**params)
    model.fit(
    
        train_x, train_y, eval_set=[(valid_x, valid_y)],
        use_best_model=True
    )
    
    # validation prediction

    preds = model.predict(valid_x)
    pred_labels = np.rint(preds)
    score = f1_score(valid_y, pred_labels)
    return score
```

### 3. optuna 진행
##### optuna.create_study()를 생성하며, f1-score를 최대로 하는 방향으로 지정한다.(direction='maximize')
##### objective 함수를 활용해 optimize를 진행하며, 반복 횟수(n_trials)는 20으로 설정한다.

```python
study = optuna.create_study(
    direction='maximize',
    study_name='CatbClf'
)

study.optimize(
    objective, 
    n_trials=20
)
```

### 4. 최적의 파라미터 값 출력
##### study.best_trial.params에 저장되어 있는 최적의 params를 Best_params로 정의한다.
##### f1_score이 최대였던 trial의 f1_score 값과 params 리스트를 각각 Best Trial, Best Params로 출력한다.

```python
Best_params = study.best_trial.params
print(f"Best Trial: {study.best_trial.value}")
print(f"Best Params: {study.best_trial.params}")
```





