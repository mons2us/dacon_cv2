## Usage:

### 주의사항
main.py의 base_dir, ckpt_path, pretrained_weights_dir는 각자 working directory 기반으로 맞출 것

dataset 폴더에
1) dirty_mnist_2nd_answer.csv와 sample_submission.csv 파일을 넣고
2) train, test 폴더 생성하여 안에 각각 이미지 파일 집어넣기

이외 폴더 구조는 동일하게 하면 됨

### TRAINING
```bash
cd {dir_to_base_folder} # git pull 한 경우 ../1_models/SKS/ 로 이동해서 아래 진행
python main.py --mode=train \
               --batch_size=32 \
               --data_type=original \ # 원본 데이터 사용할 경우 original, denoised data인 경우 denoised
               --fold_k=1 \ # fold_k=1이면 그냥 k-fold 쓰지 않는 학습이고, 2 이상인 경우 그만큼 fold 나눠서 학습 수행
               --val_ratio=0.25 \ # fold_k >= 2이면 필요없음
               --epochs=80 \
               --patience=7 \ # early stopping에 쓰이는 patience
               --verbose=100 \ # 학습 시 loss/accuracy print 해주는 주기
               --model_index=1 \ # 0보다 큰 정수값. ckpt 폴더 안에 인덱스별로 폴더 생성해서 early stopped checkpoint 저장해줌. 이미 있는 인덱스면 오류
               --learning_rate=0.002 \
               --device_index=0 \ # 0이면 cuda:0, 1이면 cuda:1
               --seed=227182 \ # 웬만하면 고정
               --base_model=plain_efficientnetb4 \ # plain_resnet50, plain_efficientnetb4, plain_efficientnetb5 ...
               --pretrained # EMNIST로 pretrain한 weight가 있는 경우에 사용, 아니면 이 argument 지우면 됨
```

### TEST
```bash
python main.py --mode=test \
               --model_index=1 \ # inference 진행하려는 model index
               --fold_k=1 \ # inference에 쓰는 모델과 동일한 fold 값으로 설정 (중요)
               --data_type=original \ # 학습한 모델과 동일하게 (중요)
               --base_model=plain_efficientnetb4 \ # 마찬가지로 학습한 모델과 동일하게... 아니면 오류남
               --device_index=1 \
               --tta # TTA를 사용할 경우는 이 argument를 명시해주면 됨. 안쓰면 TTA 안함. 현재는 [0 90 -90]도로 돌린 세개에 대해 예측해서 평균함
```
