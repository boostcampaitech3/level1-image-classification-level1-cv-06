# Mask Image Classification Competition Report

> [boostcamp AI Tech](https://boostcamp.connect.or.kr) - Level 1: CV_06 같이가조

### Results

  * **Test dataset for public leaderboard**
    * F1 score: 0.7525 / test accuracy: 79.5714%
    * Provisional standing: 20th / 48 teams
  * **Test dataset for private leaderboard**
    * F1 score: 0.7348 / test accuracy: 78.9048%
    * Final standing: 26th / 48 teams

### Task

#### Image Classification Task Specifications

  * **주어진 얼굴 사진을 다음의 세 가지 기준에 따라 18개의 class로 분류**
    * **Subtask 1: 마스크 착용 여부**
      * 제대로 착용함 / 잘못 착용함 / 착용하지 않음
    * **Subtask 2: 겉보기 성별**
      * 남성 / 여성
    * **Subtask 3: 나이대**
      * 29세 이하 / 30~59세 / 60세 이상

#### Image Dataset Specifications

  * **사진 속의 인물: 4500명**
  * **인물 당 사진: 7장**
    * 마스크를 제대로 착용한 사진: 5장
    * 마스크를 잘못 착용한 사진: 1장 / 마스크를 착용하지 않은 사진: 1장
  * **Dataset ratio**
    * Train & validation dataset: 60%
    * Test dataset for public leaderboard: 20% / test dataset for private leaderboard: 20%

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

> 가장 높은 f1 score를 달성한 모델에 대해서만 기록

  * **Model: Hard voting ensemble of 3 predictions**
    * EfficientNet_b3(trained 5 epochs / result of 2nd epoch)*
    * EfficientNet_b4(trained 4 epochs / result of 1st epoch)
    * EfficientNet_b3(trained 5 epochs / result of 5th epoch)* (\*: Single train procedure)
  * **예측 3회의 예측 class가 모두 다른 경우: 121/12600건**
    * Subtask별 예측 class마다 각각의 최빈값을 구하여 이들을 조합: 119/12600건
    * Subtask별 예측 class가 모두 다른 경우 첫 번째 모델의 것을 따름: 2/12600건
      * Class [3, 2, 1] -> Class 0
      * Class [5, 1, 0] -> Class 2
  * **Hyperparameters**
    * Optimizer: torch.optim.AdamW(weight_decay=0.001)
    * Criterion: nn.CrossEntropyLoss()
    * Learning rate: [3, 3, 2, 2, 1]e-4 for EfficientNet_b3 / [8, 5, 3, 2]e-4 for EfficientNet_b4
    * Batch size: 40 for EfficientNet_b3 / 32 for EfficientNet_b4
  * **Data augmentation**
    * transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  * **Train & validation dataset split rule**
    * 정답 class별로 8:2 split을 진행하여 양쪽의 class 분포를 맞춤
    * 같은 사람의 사진은 모두 한쪽으로만 들어가도록 split 진행(data leakage 예방)

### Thougts

> 다들 처음 진행해보는 대회여서 적응하는 시간이 필요했지만, 성공적으로 마무리한 것 같다. <br><br>
> 우리는 강의에서 배운 EDA, data augmentation, modeling 등을 차례대로 수행해 나갔고, mixup과 GAN 등의 색다른 시도들도 해 보았다. 그리고 다양한 모델(ResNet, EfficientNet, ReXNet, VGGNet 등)들을 가져와 직접 hyperparameter tuning을 진행하며 실험해 보았고, ensemble도 적용해 보았다. 마지막으로는 베이스라인 코드의 대부분을 그대로 사용하지 않고 수정하며, 모델 구현에 대한 자신감을 가질 수 있었다. 이 과정에서 많은 에러가 발생했지만, 슬랙과 노션으로 서로 공유하면서 팀원들과 함께 해결해 나갔다. <br><br>
> 물론 아쉬움도 많이 남는다. 우수 사례 발표를 들어 보면서, 저런 방법도 있구나 하는 생각을 많이 하게 되었다. 특히 체계적인 설계에 의한 프로젝트 진행이 돋보였다. 우리도 더욱 다양한 실험을 했더라면 더 좋았을 것이고, 실험의 성과를 정량적으로 분석하지 못한 점과, test score가 validation score보다 일관적으로 낮게 나오는 이유를 데이터 분포 분석 등의 방법으로 알아내서 활용하지 못한 점이 아쉬웠다. <br><br>
> 그래도 다음에는 우리도 더 잘 할 수 있을 것이라고 생각한다. 이번 경험을 통해 얻은 긍정적인 반성들을 되새기며 다음 프로젝트를 진행한다면 앞으로 더욱 더 좋은 성과가 있을 것이다. 다음 스테이지가 정말 기대된다.

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
