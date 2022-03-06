# Mask Image Classification Competition Report

> [boostcamp AI Tech](https://boostcamp.connect.or.kr) - Level 1: CV_06 같이가조

### Results

  * **Test dataset for public dataset**
    * F1 score: 0.7525
    * Test accuracy: 79.5714%
    * Provisional standing: 20th / 48 teams
  * **Test dataset for private dataset**
    * F1 score: 0.7348
    * Test accuracy: 78.9048%
    * Final standing: 26th / 48 teams

### Task

#### Image Classification Task Specifications

  * **주어진 얼굴 사진을 다음의 세 가지 기준에 따라 18개의 class로 분류**
    * **마스크 착용 여부**
      * 제대로 착용함
      * 잘못 착용함
      * 착용하지 않음
    * **겉보기 성별**
      * 남성
      * 여성
    * **나이대**
      * 29세 이하
      * 30~59세
      * 60세 이상

#### Image Dataset Specifications

  * **사진 속의 인물: 4500명**
  * **인물 당 사진: 7장**
    * 마스크를 제대로 착용한 사진: 5장
    * 마스크를 잘못 착용한 사진: 1장
    * 마스크를 착용하지 않은 사진: 1장
  * **Dataset ratio**
    * Train & validation dataset: 60%
    * Test dataset for public leaderboard: 20%
    * Test dataset for private leaderboard: 20%

#### Main Difficulties

  * **Data imbalance**
    * 겉보기 성별이 남성인 사진과 나이대가 60세 이상인 사진의 비율이 유의미하게 낮았음
  * **Label noise**
    * Train & validation dataset에 잘못 분류된 사진이 209/18900장(1.11%) 있었음
  * **Subtask cross dependency**
    * Subtask 3개 간의 상호 dependency로 인해 class 18개의 독립분포 가정이 위배됨

### Approaches

  * **Dealing with data imbalance**
    * 실험 대상 모델을 고를 때 후보 모델들의 data imbalance robustness를 고려함
    * GAN 등의 색다른 모델에 대한 실험도 진행해 봄
    * F1 loss 사용 실험 진행
  * **Dealing with label noise**
    * ResNet-18을 10번 새로 학습시켜 얻은 10개의 validation dataset class 예측 중에서 2회 이상 틀린 경우를 전수조사하여 train & validation dataset의 image 209장의 label 교정
  * **Dealing with subtask cross dependency**
    * Generalization 성능이 좋은 AdamW optimizer 사용
    * Image mixup 및 crop 여부에 따른 성능 비교 실험 진행
    * Soft voting 내지는 hard voting ensemble 적극 활용

### Technical Specifications

  * **Model: Hard voting ensemble of 3 predictions**
    * EfficientNet_b3(trained 5 epochs / result of 2nd epoch)*
    * EfficientNet_b4(trained 4 epochs / result of 1st epoch)
    * EfficientNet_b3(trained 5 epochs / result of 5th epoch)*
  * **예측 3회의 예측 class가 모두 다른 경우: 121/12600건**
    * 예측 3회의 mask, gender, age subtask별 예측 class 각각의 최빈값을 구하여 이들을 조합한 class를 따름: 119/12600건
    * 예측 3회에서 age subtask 예측 class까지 다른 경우 age subtask 예측 class는 첫 번째 예측의 것을 따름: 2/12600건
      * Class [3, 2, 1] -> Class 0
      * Class [5, 1, 0] -> Class 2
  * **Hyperparameters**
    * Optimizer: torch.optim.AdamW(weight_decay=0.001)
    * Criterion: nn.CrossEntropyLoss()
    * Learning rate
      * For EfficientNet_b3: [0.0003, 0.0003, 0.0002, 0.0002, 0.0001]
      * For EfficientNet_b4: [0.0008, 0.0005, 0.0003, 0.0002]
    * Batch size
      * For EfficientNet_b3: 40
      * For EfficientNet_b4: 32
  * **Data augmentation**
    * transforms.ToTensor()
    * transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  * **Train & validation dataset split rule**
    * class ProfileClassEqualSplitTrainMaskDataset
    * 정답 class별로 8:2 split을 진행하여 양쪽의 class 분포를 맞춤
    * 같은 사람의 사진은 class와 무관하게 한쪽으로만 들어가도록 split 진행
    * Data leakage 예방

\*: Single train procedure

### Lessons

#### Failures

> To be written <br>
> 실패했던 지점들 작성 예정

#### Future Directions

> To be written <br>
> 도입 가능했을 개선 방향들 작성 예정

### Thougts

> To be written <br>
> 전반적인 대회 소감 작성 예정

### How To Train

1. Execute main code to train EfficientNet_b3

```shell
python main.py
```

2. Change model instance, batch size and learning rate
3. Execute fixed main code to train EfficientNet_b4

```shell
python main.py
```
