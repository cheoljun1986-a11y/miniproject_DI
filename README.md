# FFC 2단계 복원 파이프라인 (jaehoon)

최종 과제(번짐 + 잡음이 섞인 영상에서 원본 영상을 복원)를 **두 단계로 나눠** 푼다.
두 단계 모두 FFC(Fast Fourier Convolution) 기반 U-Net 이고, 같은 구조를 쓴다.

```
번짐+잡음 영상 ──▶ [1단계: 잡음 제거] ──▶ 번진 영상 ──▶ [2단계: 역합성곱] ──▶ 원본 영상
   18.51 dB          train_denoise_ffc        30.27 dB       train_deconv_ffc        최종 점수
```

FFC 는 채널을 둘로 나눠 절반은 보통의 3×3 합성곱(국소 갈래)으로, 절반은
FFT → 1×1 합성곱 → 역FFT(전역 갈래)로 처리하는 블록이다. 전역 갈래 덕분에
한 층이 영상 전체를 보므로, 번짐(다이폴 커널과의 합성곱)처럼 멀리 퍼지는
현상을 다루기에 알맞다.

## 현재 성적

| 무엇 | 점수 | 비고 |
|---|---|---|
| 1단계 (번짐+잡음 → 번진 영상, test 100장) | **30.266 dB / SSIM 0.9376** | 학습 완료 (200 epoch) |
| 2단계 포함 전체 (번짐+잡음 → 원본, val) | **25.874 dB @ ep 45** | 학습 진행 중 (960 epoch 예정) |
| 조교 end-to-end U-Net (비교 대상) | 25.02 dB | **ep 24 에 추월** |

2단계 학습 로그는 이 문서 맨 아래에 있다.

## 파일

| 파일 | 내용 |
|---|---|
| `train_denoise_ffc.ipynb` | **1단계.** 번짐+잡음 영상에서 잡음만 지워 번진 영상을 만든다 |
| `train_deconv_ffc.ipynb` | **2단계.** 번진 영상에서 번짐을 되돌려 원본 영상을 만든다. 1단계 체크포인트를 불러 쓴다 |
| `train_restormer.ipynb` | 1일차(번짐 없음) 조건의 Restormer 잡음 제거. test 35.677 dB |
| `a100_denoising.py`, `train_denoising_a100.ipynb` | junsung 님 DnCNN 미세조정 코드 (main 에서 병합됨) |
| `checkpoint_best_a100*.ckpt` | DnCNN 체크포인트 |

## 1단계 — train_denoise_ffc.ipynb

- **학습 쌍 만들기**: 깨끗한 학습 영상을 다이폴 커널로 번지게 한 뒤(6개 방향 중
  무작위) 잡음을 섞는다. 입력 = 번진 영상 + 잡음, 정답 = 번진 영상.
- **잡음 4종**: gaussian, rician, uniform, salt_and_pepper. 세기는 종류별 범위에서
  무작위. rician 이 유독 약해서 학습에서 40% 비중으로 더 자주 뽑는다.
- **손실**: L1 + 국소 평균 항(8/16/32 창의 평균끼리 L1). rician 잡음은 밝기를
  위로 미는 편향(+0.029)이 있는데 L1 만으로는 이 편향을 못 잡아서 넣었다.
- **전역 skip**: 입력과 출력이 거의 같은 문제라 `출력 = 신경망(입력) + 입력` 꼴을 쓴다.

test 100장 종류별 성적:

| 잡음 | 입력 | FFC U-Net | Mean 3×3 | Median 3×3 |
|---|---|---|---|---|
| gaussian | 19.953 | **31.029** | 22.801 | 22.627 |
| rician | 14.557 | **23.229** | 17.024 | 16.900 |
| uniform | 22.295 | **31.610** | 23.499 | 22.445 |
| salt_and_pepper | 17.223 | **35.198** | 22.359 | 26.698 |
| 전체 | 18.507 | **30.266** | 21.421 | 22.167 |

남은 약점: rician (전체 제곱오차의 72.7%를 혼자 차지). rician 을 나머지 수준까지
올리면 전체가 +2.3 dB 오른다.

## 2단계 — train_deconv_ffc.ipynb

핵심 설계 세 가지:

1. **학습 입력 = 얼린 1단계의 실제 출력** (`STAGE1_MODE = "frozen"`).
   시험 때 2단계가 받는 것은 1단계 출력이므로 학습 입력도 같은 분포여야 한다.
   역합성곱은 입력 분포의 어긋남을 최대 44,074배까지 증폭하기 때문에 중요하다.
2. **다이폴 지도 주입** (`USE_DIPOLE_FEATURE = True`). FFC 전역 갈래의 1×1
   합성곱은 모든 주파수에 같은 가중치를 써서, 주파수마다 다른 이득 1/D(k)를
   표현할 수 없다. 각 주파수 자리의 D, log|D|, sign(D) 세 채널을 함께 넣어
   이 한계를 풀었다. 조교 공지의 "물리 기반 모델에 다이폴 모델 사용 가능"에 해당.
3. **방향 고정** (`FIX_ORIENTATION = True`). 시험은 B0 = (0, 1) 한 방향이므로
   학습도 그 방향만 쓴다. 6개 방향에 용량을 흩지 않는다.
   ※ 최종 시험이 정말 한 방향인지는 조교 확인이 필요하다. 여러 방향이면
   이 스위치를 꺼야 한다.

보조 장치:

- **물리 항**: 손실에 `0.1 × |dipole(복원 결과) − 깨끗한 번진 영상|` 추가.
- **발산 감지기**: loss 가 직전의 5배를 넘으면 best 체크포인트로 되돌리고
  학습률을 절반으로 줄여 계속 간다.
- **EPOCH_FRACTION = 4**: 한 epoch 에 학습 영상의 1/4만 무작위로 쓴다. 총
  계산량은 같지만 epoch 수가 늘어 코사인 학습률 곡선이 끝까지 내려간다.
- `STAGE1_MODE = "e2e"` 로 바꾸면 1단계 없이 한 모델이 번짐+잡음 → 원본을
  통째로 배우는 대조 실험이 된다 (조교 U-Net 과 같은 구도).

## 실행 방법 (Colab)

1. Drive 의 `MyDrive/project5/` 에 `dataset.zip` 을 둔다 (2단계는 이것만 필요).
2. `train_denoise_ffc.ipynb` 를 먼저 끝낸다 → `logs_denoise_ffc_blur/checkpoint_best.ckpt`
   가 Drive 에 생긴다.
3. **새 런타임에서** `train_deconv_ffc.ipynb` 를 위에서부터 실행한다.
   (1단계 체크포인트가 없으면 2단계는 시작하지 못하고 명확한 오류를 낸다.)
4. 체크포인트는 매 epoch 로컬에, 30 epoch 마다 Drive 에 저장된다. 끊겨도
   노트북을 다시 실행하면 이어서 학습한다.

## 규칙 관련 주의

- **Wiener 필터는 최종 제출에 사용 금지.** 노트북 안의 TKD/Wiener 코드는
  파이프라인 부품이 아니라 참고용 계기판이다 (우리 결과와 나란히 찍어
  고전 방법 대비 위치를 보는 용도).
- **test set 은 점수 확인 외 사용 금지.** 모든 관찰·튜닝은 val 에서 한다.
- 지도학습(제공된 깨끗한 영상 사용)은 명시적으로 허용됨.

## 2단계 학습 로그 (v2, epoch 1–49)

방향 고정 + 다이폴 지도를 넣기 전(v1)에는 ep20 에 17.681 이었다. 두 수정 뒤
같은 지점에서 +6.8 dB:

```
ep   1/960  loss 0.13469  val PSNR  18.939  SSIM 0.5674  [0.7 min]  <- best
ep   2/960  loss 0.09045  val PSNR  18.912  SSIM 0.6172  [1.1 min]
ep   3/960  loss 0.09139  val PSNR  20.395  SSIM 0.6506  [1.6 min]  <- best
ep   4/960  loss 0.08162  val PSNR  20.441  SSIM 0.6640  [2.1 min]  <- best
ep   5/960  loss 0.07897  val PSNR  19.522  SSIM 0.6716  [2.6 min]
ep   6/960  loss 0.07771  val PSNR  21.348  SSIM 0.6880  [3.1 min]  <- best
ep   7/960  loss 0.07011  val PSNR  18.564  SSIM 0.6844  [3.5 min]
ep   8/960  loss 0.06412  val PSNR  22.002  SSIM 0.7017  [4.0 min]  <- best
ep   9/960  loss 0.06571  val PSNR  22.853  SSIM 0.7203  [4.5 min]  <- best
ep  10/960  loss 0.05807  val PSNR  22.065  SSIM 0.7211  [5.0 min]
ep  11/960  loss 0.05880  val PSNR  21.796  SSIM 0.7148  [5.5 min]
ep  12/960  loss 0.05690  val PSNR  23.155  SSIM 0.7264  [6.0 min]  <- best
ep  13/960  loss 0.05437  val PSNR  23.912  SSIM 0.7372  [6.4 min]  <- best
ep  14/960  loss 0.05682  val PSNR  22.544  SSIM 0.7261  [6.9 min]
ep  15/960  loss 0.05345  val PSNR  23.779  SSIM 0.7391  [7.4 min]
ep  16/960  loss 0.04857  val PSNR  23.196  SSIM 0.7309  [7.9 min]
ep  17/960  loss 0.04815  val PSNR  24.437  SSIM 0.7507  [8.4 min]  <- best
ep  18/960  loss 0.05037  val PSNR  23.136  SSIM 0.7409  [8.8 min]
ep  19/960  loss 0.04672  val PSNR  21.652  SSIM 0.7371  [9.3 min]
ep  20/960  loss 0.04570  val PSNR  24.476  SSIM 0.7576  [9.8 min]  <- best
ep  21/960  loss 0.04581  val PSNR  24.709  SSIM 0.7546  [10.3 min]  <- best
ep  22/960  loss 0.04317  val PSNR  24.727  SSIM 0.7567  [10.8 min]  <- best
ep  23/960  loss 0.04518  val PSNR  24.754  SSIM 0.7621  [11.3 min]  <- best
ep  24/960  loss 0.04341  val PSNR  25.300  SSIM 0.7641  [11.7 min]  <- best
ep  25/960  loss 0.04231  val PSNR  24.709  SSIM 0.7604  [12.2 min]
ep  26/960  loss 0.04216  val PSNR  23.895  SSIM 0.7544  [12.7 min]
ep  27/960  loss 0.04164  val PSNR  25.316  SSIM 0.7691  [13.2 min]  <- best
ep  28/960  loss 0.04173  val PSNR  25.033  SSIM 0.7563  [13.7 min]
ep  29/960  loss 0.03932  val PSNR  24.762  SSIM 0.7610  [14.1 min]
ep  30/960  loss 0.04122  val PSNR  25.013  SSIM 0.7696  [14.6 min]
ep  31/960  loss 0.03944  val PSNR  24.726  SSIM 0.7717  [15.1 min]
ep  32/960  loss 0.03959  val PSNR  25.225  SSIM 0.7731  [15.6 min]
ep  33/960  loss 0.03831  val PSNR  23.239  SSIM 0.7712  [16.1 min]
ep  34/960  loss 0.04042  val PSNR  24.759  SSIM 0.7569  [16.6 min]
ep  35/960  loss 0.03864  val PSNR  25.681  SSIM 0.7741  [17.0 min]  <- best
ep  36/960  loss 0.03861  val PSNR  25.742  SSIM 0.7761  [17.5 min]  <- best
ep  37/960  loss 0.03817  val PSNR  25.237  SSIM 0.7726  [18.0 min]
ep  38/960  loss 0.03907  val PSNR  25.374  SSIM 0.7768  [18.5 min]
ep  39/960  loss 0.03880  val PSNR  25.823  SSIM 0.7771  [19.0 min]  <- best
ep  40/960  loss 0.03845  val PSNR  24.316  SSIM 0.7694  [19.4 min]
ep  41/960  loss 0.03855  val PSNR  24.901  SSIM 0.7746  [19.9 min]
ep  42/960  loss 0.03863  val PSNR  23.553  SSIM 0.7727  [20.4 min]
ep  43/960  loss 0.03816  val PSNR  24.624  SSIM 0.7780  [20.9 min]
ep  44/960  loss 0.03799  val PSNR  25.578  SSIM 0.7748  [21.4 min]
ep  45/960  loss 0.03820  val PSNR  25.874  SSIM 0.7807  [21.8 min]  <- best
ep  46/960  loss 0.03719  val PSNR  25.183  SSIM 0.7750  [22.3 min]
ep  47/960  loss 0.03635  val PSNR  25.222  SSIM 0.7751  [22.8 min]
ep  48/960  loss 0.03893  val PSNR  24.886  SSIM 0.7775  [23.3 min]
ep  49/960  loss 0.03915  val PSNR  25.690  SSIM 0.7787  [23.8 min]
```

이 val PSNR 은 원본 영상과 비교한 값, 즉 **과제 전체의 점수**다 (번짐+잡음
영상이 들어가 1단계와 2단계를 모두 통과한 결과를 원본과 비교). 조교
end-to-end U-Net 의 25.02 와 같은 기준이다.
