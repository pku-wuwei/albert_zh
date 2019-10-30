#!/usr/bin/env bash

BERT_BASE_DIR=./albert_config
python3 run_pretraining.py \
 --bert_config_file=$BERT_BASE_DIR/albert_config_base.json \
 --input_file=data/tf_news_2016_zh_raw_news2016zh_1.tfrecord \
 --output_dir=data/ \
 --do_train=true \
 --do_eval=true \
 --use_tpu=true \
 --do_whole_word_mask=True \
 --input_file=data/news_zh_1.txt \
 --vocab_file=$BERT_BASE_DIR/vocab.txt \
 --do_lower_case=True \
 --max_seq_length=512 \
 --max_predictions_per_seq=51 \
 --masked_lm_prob=0.10
