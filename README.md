# Mask Image Classification Competition Report

> [boostcamp AI Tech](https://boostcamp.connect.or.kr) - Level 1: CV_06 같이가조 

### Results

  * Public test dataset
    * F1 score: 0.7525
    * Test accuracy: 79.5714%
    * Provisional standing: 20th / 48 teams
  * Private test dataset
    * F1 score: 0.7348
    * Test accuracy: 78.9048%
    * Final standing: 26th / 48 teams

### Task

#### Image Classification Task Specifications

> To be written <br>
> 대회 개요 작성 

#### Main Difficulties

> To be written <br>
> 어려웠던 지점들 작성 예정

### Approaches

> To be written <br>
> 프로젝트 데이터 정제 과정 작성 예정 <br>
> 실험해 본 내역 작성 예정

### Technical Specifications

  * Model: Hard voting ensemble of 3 predictions
    * EfficientNet_b3(trained 5 epochs / result of 2nd epoch)*
    * EfficientNet_b4(trained 4 epochs / result of 1st epoch)
    * EfficientNet_b3(trained 5 epochs / result of 5th epoch)*
  * Conflict resolving rule: 예측 3회의 예측 class가 모두 다른 경우
    * 121/12600건
    * 예측 3회의 mask, gender, age subtask별 예측 class 각각의 최빈값을 구하여 이들을 조합한 class를 따름
      * 119/12600건
    * 예측 3회에서 age subtask 예측 class까지 다른 경우 age subtask 예측 class는 첫 번째 예측의 것을 따름
      * 2/12600건
      * Class [3, 2, 1] -> Class 0
      * Class [5, 1, 0] -> Class 2
  * Hyperparameters
    * Optimizer: torch.optim.AdamW(weight_decay=0.001)
    * Criterion: nn.CrossEntropyLoss()
    * Learning rate
      * For EfficientNet_b3: [0.0003, 0.0003, 0.0002, 0.0002, 0.0001]
      * For EfficientNet_b4: [0.0008, 0.0005, 0.0003, 0.0002]
    * Batch size
      * For EfficientNet_b3: 40
      * For EfficientNet_b4: 32
  * Data augmentation
    * transforms.ToTensor()
    * transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  * Train & validation dataset split rule
    * 정답 class별로 8:2 split을 진행하여 양쪽의 class 분포를 맞춤
    * 같은 사람의 사진은 class와 무관하게 한쪽으로만 들어가도록 split 진행

\* Single train procedure

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
