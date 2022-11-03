# L2R2 + RoBERTa & DeBERTa

This repository is part of [the project CL-Teamlab SS21](https://github.com/esradonmez/CL-teamlab) by Esra Dönmez and Nianheng Wu. It contains the Learn2Rank part of implementation. We adapted the code, so it can run with [DeBERTa](https://arxiv.org/pdf/2006.03654.pdf) (in addition to its original ability of running RoBERTa).

The motivation of using learn2rank framework is in the README part of [the main page of this project](https://github.com/RealNicolasBourbaki/Learn-to-rank-for-Abductive-Reasoning).

## Usage

### Choose the right model

This project support RoBERTa and DeBERTa pretrained model. Go to the corresponding sub-folder after you decided on which model to use.

### Set up environment

This project has been tested on Python 3.8 with PyTorch 1.4.0.

We recommend you to create virtual environment for running the code.

**Reminder**: the dependencies for running DeBERTa and RoBERTa are slightly different.

```shell script
$ pip install -r requirements.txt
```

### Prepare data

Get the dataset released by [αNLI](https://leaderboard.allenai.org/anli/submissions/get-started)
```shell script
$ wget https://storage.googleapis.com/ai2-mosaic/public/alphanli/alphanli-train-dev.zip
$ unzip -d alphanli alphanli-train-dev.zip
```

### Training

The original code use all GPU for training. In order to avoid taking too much public resource from IMS, we changed the code to only run on one GPU. The changes could be found in ```run.py``` with comment suggesting the changes.

The available `criterion` for optimization could selected in:
- list_net: list-wise *KLD* loss used in ListNet
- list_mle: list-wise *Likelihood* loss used in ListMLE
- approx_ndcg: list-wise *ApproxNDCG* loss used in ApproxNDCG
- rank_net: pair-wise *Logistic* loss used in RankNet
- hinge: pair-wise *Hinge* loss used in Ranking SVM
- lambda: pair-wise *LambdaRank* loss used in LambdaRank

Note that in our experiment, we manually reduce the learning rate instead of using any automatic learning rate scheduler.

For example, we first fine-tune the pre-trained RoBERTa-large model for up to 10 epochs with a learning rate of 5e-6 and save the model checkpoint which performs best on the dev set.
```shell script
$ CUDA_VISIBLE_DEVICES=[N] python run.py \
  --data_dir=[where you store the datasets]/ \
  --output_dir=ckpts/ \
  --model_type='roberta' \ # or 'deberta'
  --model_name_or_path='roberta-large' \ # or 'microsoft/deberta-large'
  --linear_dropout_prob=0.6 \
  --max_hyp_num=22 \
  --tt_max_hyp_num=22 \
  --max_seq_len=72 \
  --do_train \
  --do_eval \
  --criterion='list_net' \
  --per_gpu_train_batch_size=1 \
  --per_gpu_eval_batch_size=1 \
  --learning_rate=5e-6 \
  --weight_decay=0.0 \
  --num_train_epochs=10 \
  --seed=42 \
  --log_period=50 \
  --eval_period=100 \
  --overwrite_output_dir
```

Then, we continue to fine-tune the just saved model for up to 3 epochs with a smaller learning rate, such as 3e-6, 1e-6 and 5e-7, until the performance on the dev set is no longer improved.
```shell script
python run.py \
  --data_dir=alphanli/ \
  --output_dir=ckpts/ \
  --model_type='roberta' \
  --model_name_or_path=ckpts/H22_L72_E3_B4_LR5e-06_WD0.0_MMddhhmmss/checkpoint-best_acc/ \
  --linear_dropout_prob=0.6 \
  --max_hyp_num=22 \
  --tt_max_hyp_num=22 \
  --max_seq_len=72 \
  --do_train \
  --do_eval \
  --criterion='list_net' \
  --per_gpu_train_batch_size=1 \
  --per_gpu_eval_batch_size=1 \
  --learning_rate=1e-6 \
  --weight_decay=0.0 \
  --num_train_epochs=3 \
  --seed=43 \
  --log_period=50 \
  --eval_period=100 \
  --overwrite_output_dir
```
Note: change the seed to reshuffle training samples.

### Evaluation

Evaluate the performance on the dev set.
```shell script
$ export MODEL_DIR="ckpts/H22_L72_E3_B4_LR5e-07_WD0.0_MMddhhmmss/checkpoint-best_acc/"
$ python run.py \
  --data_dir=alphanli/ \
  --output_dir=$MODEL_DIR \
  --model_type='roberta' \
  --model_name_or_path=$MODEL_DIR \
  --max_hyp_num=2 \
  --max_seq_len=72 \
  --do_eval \
  --per_gpu_eval_batch_size=1
```


