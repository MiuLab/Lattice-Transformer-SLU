Adapting Pretrained Transformer to Lattices for Spoken Language Understanding
===
[Paper](https://www.csie.ntu.edu.tw/~yvchen/doc/ASRU19_LatticeSLU.pdf)

This repo contains source code of our ASRU 2019 paper "*Adapting Pretrained Transformer to Lattices for Spoken Language Understanding*"


## Requirements
* Python >= 3.6

Required python packages are listed in *requirements.txt*.

## Dataset
Unfortunately, we are not allowed to redistribute the dataset(ATIS). The dataset needs to be obtained from LDC
* https://catalog.ldc.upenn.edu/LDC93S5
* https://catalog.ldc.upenn.edu/LDC94S19
* https://catalog.ldc.upenn.edu/LDC95S26

## Preprocess

#### Convert lattices to PLF format
We use the PLF format lattices, you can use this script to convert Kaldi lattices to PLF format
> https://github.com/noisychannel/phrase_speech_translation/blob/master/asr_util/kaldi2FST.sh

#### Create dataset
    python3 preproc-lattice.py [-h] dataset_file lattice_file out_file

* **dataset_file**: csv file with fields *id*, *text*, *labels*. The *id* field should match with the utterance ids.
* **lattice_file**: PLF lattice generated from the above script.
* **out_file**: output filename.

## Training
Sample usage:

    python3 run_openai_gpt_atis_lattice.py
        --train_dataset <train_csv_file>
        --eval_dataset <eval_csv_file>
        --model_name openai-gpt
        --output_dir <output_dir>
        --do_train --do_eval
        --task <intent/slot>
        --num_train_epochs 5
        --attn_bias
        --probabilistic_masks

* **probabilistice_masks**: Whether to use probabilistic_masks. Binary masks will be used if not set.
* **linearize**: linearize lattices.

## Reference
Please cite the following paper

    @inproceedings{
        huang2019adapting,
        title={Adapting Pretrained Transformer to Lattices for Spoken Language Understanding},
        author={Chao-Wei Huang and Yun-Nung Chen},
        booktitle={2019 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)},
        year={2019},
        organization={IEEE}
    }
