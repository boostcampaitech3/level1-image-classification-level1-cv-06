# level1-image-classification-level1-cv-06
level1-image-classification-level1-cv-06 created by GitHub Classroom

### Mixup Review

요약 : Mixup은 한정적인 학습 데이터를 통해 cost를 최소화할 수 있는 data augmentation 방법론이다. 다른 샘플 데이터 i번째와 j번째 $(x_i, x_j)$를 $$\alpha$$ 비율만큼 섞어서 새로운 데이터 $$\tilde{x}$$ 를 생성한다. 물론 이때 label도 $$\alpha$$ 만큼 섞어서 사용한다. 

![mixup1](https://user-images.githubusercontent.com/44994262/156910562-d8f7d4f6-ef2f-4237-a81f-5182e8709817.png)

$x_i$와 $x_j$ 선형결합을 통해 새로운 데이터 $\tilde{x}$를 얻을 수 있다. 

![mixup2](https://user-images.githubusercontent.com/44994262/156910574-9ad98b88-4ffe-423b-900f-848383cdb16a.png)

예시 : 고양이와 강아지를 0.5씩 섞었다.

![mixup3](https://user-images.githubusercontent.com/44994262/156910583-447246ff-4b64-4bfc-a489-d4ed347a7e1a.png)

위 코드를 보면, data load하는 부분에서 원래 1개의 data를 불러오는 것을 2개 불러오는 식으로만 변경하면 된다. 기존 코드에 추가하기 편리할 것으로 예상한다. 다만 라벨(y)가 one-hot vector여야 한다.

코드에서 살펴보면, $x_1$과 $x_2$ 이미지가 있고, 각각의 label이 $y_1$, $y_2$이다. 이를 mixup해서 새로운 데이터 $x,y$를 얻을 것이다. lam 만큼의 비율로 섞을 것인데, 이때 이 lam은 **베타분포**를 따른다. 

![mixup4](https://user-images.githubusercontent.com/44994262/156910588-19b6e43f-ff51-4f47-bf71-4a818371a391.png)

우선 베타 분포는 확률에 대한 확률분포이다. 예를들어 고양이, 강아지가 2:8로 mixup된 이미지를 보고 고양이를 선택할 확률이 0.5이상일 확률을 구할 때 베타분포를 사용할 수 있다. 
이미 학습과정에서 3번은 고양이, 10번은 강아지로 분류하였다면, 현재 분류자가 고양이를 선택할 확률이 0.5보다 클 확률은 0.02이다. 

![mixup5](https://user-images.githubusercontent.com/44994262/156910596-951228d9-64e7-4d6e-a434-598102283944.png)

여기서 3과 10은 각각 실패횟수와 성공횟수를 의미하는데, 이것이 1보다 작아지면 어떻게 해석을 해야할지 조금 난감했다. 

![mixup6](https://user-images.githubusercontent.com/44994262/156910607-7395f070-9d7e-49d7-8527-fdce07f53a71.png)


본 논문에서는 $\alpha$ 값으로 0.1 ~ 0.4 사이의 값을 추천한다. $\alpha$ 값이 커지면 underfitting에 이른다고 설명한다. 람다는 Beta($\alpha$ , $\alpha$ )분포를 따르고, 실제 $\alpha$ 값으로 0.1~0.4를 권장하므로, 베타 분포는 위와 같은 모습을 보인다. 

U자 형태인데*(빨간 선 주목)*, 이는 두 이미지를 섞을 때 강아지, 고양이 그림 예시처럼 0.5, 0.5 섞는 것보다, 아주 조금의 다른 이미지를 섞어서 사용한다는 것을 알 수 있었다. 

실험 결과는 다음과 같다. 

![mixup7](https://user-images.githubusercontent.com/44994262/156910626-eaf4aaba-c2b6-4995-a8c5-d8f00129b6ea.png)

위 표는 라벨이 잘못붙어있을 때, mixup을 사용했다. 실제 데이터는 라벨이 잘못 붙어있는 경우도 존재한다. 각각 라벨의 20%, 50%, 80%가 잘못붙어있을 때의 결과인데 mixup을 사용했을 때 error가 많이 줄어듦을 알 수 있다. 
mixup을 사용하게되면, 두 이미지를 섞어주니 라벨이 잘못 붙어있어도, 이미지를 잘 분류할 확률이 높아지는 것이라고 생각할 수 있겠다.
