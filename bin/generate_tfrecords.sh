#!/bin/bash
DATASET_DIR=../VOC2012/
OUTPUT_DIR=../tfrecords
python tf_convert_data.py --dataset_name=pascalvoc --dataset_dir=${DATASET_DIR} --output_name=voc_2012_train --output_dir=${OUTPUT_DIR}
