<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
## 요약 : Mixup은 한정적인 학습 데이터를 통해 cost를 최소화할 수 있는 data augmentation 방법론이다. 다른 샘플 데이터 i번째와 j번째 $(x_i, x_j)$ 를 $\alpha$ 비율만큼 섞어서 새로운 데이터 $\tilde{x}$ 를 생성한다. 물론 이때 label도 $\alpha$ 만큼 섞어서 사용한다. 

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/27f69f79-f315-48be-8a64-67e7205aa272/Untitled.png)

$x_i$와 $x_j$ 선형결합을 통해 새로운 데이터 $\tilde{x}$를 얻을 수 있다. 

![예시 : 고양이와 강아지를 0.5씩 섞었다.](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e8a3e71c-c56d-479f-b11f-5f1844c917b4/Untitled.png)

예시 : 고양이와 강아지를 0.5씩 섞었다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f87c2119-561e-4c65-81ba-f7f6ff8a7f4a/Untitled.png)

위 코드를 보면, data load하는 부분에서 원래 1개의 data를 불러오는 것을 2개 불러오는 식으로만 변경하면 된다. 기존 코드에 추가하기 편리할 것으로 예상한다. 다만 라벨(y)가 one-hot vector여야 한다.

코드에서 살펴보면, $x_1$과 $x_2$ 이미지가 있고, 각각의 label이 $y_1$, $y_2$이다. 이를 mixup해서 새로운 데이터 $x,y$를 얻을 것이다. lam 만큼의 비율로 섞을 것인데, 이때 이 lam은 **베타분포**를 따른다. 

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a172fcf5-0f21-43ad-b7a2-407ec7078b98/Untitled.png)

우선 베타 분포는 확률에 대한 확률분포이다. 예를들어 고양이, 강아지가 2:8로 mixup된 이미지를 보고 고양이를 선택할 확률이 0.5이상일 확률을 구할 때 베타분포를 사용할 수 있다. 
이미 학습과정에서 3번은 고양이, 10번은 강아지로 분류하였다면, 현재 분류자가 고양이를 선택할 확률이 0.5보다 클 확률은 0.02이다. 

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1374f21b-5a30-4524-9322-898feaaf697f/Untitled.png)

여기서 3과 10은 각각 실패횟수와 성공횟수를 의미하는데, 이것이 1보다 작아지면 어떻게 해석을 해야할지 조금 난감했다. 

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/49c0ff54-097c-447a-b334-07472ca5996c/Untitled.png)

본 논문에서는 $\alpha$ 값으로 0.1 ~ 0.4 사이의 값을 추천한다. $\alpha$ 값이 커지면 underfitting에 이른다고 설명한다. 람다는 Beta($\alpha$ , $\alpha$ )분포를 따르고, 실제 $\alpha$ 값으로 0.1~0.4를 권장하므로, 베타 분포는 위와 같은 모습을 보인다. 

U자 형태인데*(빨간 선 주목)*, 이는 두 이미지를 섞을 때 강아지, 고양이 그림 예시처럼 0.5, 0.5 섞는 것보다, 아주 조금의 다른 이미지를 섞어서 사용한다는 것을 알 수 있었다. 

실험 결과는 다음과 같다. 

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5ef3bf85-6751-4762-bec2-ccd147ebd0e0/Untitled.png)

위 표는 라벨이 잘못붙어있을 때, mixup을 사용했다. 실제 데이터는 라벨이 잘못 붙어있는 경우도 존재한다. 각각 라벨의 20%, 50%, 80%가 잘못붙어있을 때의 결과인데 mixup을 사용했을 때 error가 많이 줄어듦을 알 수 있다. 
mixup을 사용하게되면, 두 이미지를 섞어주니 라벨이 잘못 붙어있어도, 이미지를 잘 분류할 확률이 높아지는 것이라고 생각할 수 있겠다.
