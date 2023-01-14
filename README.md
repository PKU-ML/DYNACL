# [ICLR23] Rethinking the Effect of Data Augmentation in Adversarial Contrastive Learning

Authors: Rundong Luo*, Yifei Wang*, Yisen Wang from Peking University


## Introduction
---



## Environment
---
A standard pytorch environment with basic packages (e.g., numpy, pickle) is enough. To evaluate under the Auto-Attack benchmark, the [autoattack](https://github.com/fra31/auto-attack) package is required. Run the following code to install:

    pip install git+https://github.com/fra31/auto-attack

## Data
---
CIFAR10, CIFAR100, and STL10 dataset are required. You may manually download them and put them in the ``./data`` folder, or directly run our provided scripts to automatically download these datasets.

## Training
---

We assume you have a GPU with no less than 16GB memory (e.g., 3090). Evaluation require fewer memory (less than 8GB).

For DynACL, run

    python train_DynACL.py --experiment EXP_PATH --dataset DATASET_NAME --data PATH_TO_DATASET

All training results (checkpoints and loggers) will be stored in ``./experiments/EXP_PATH``. Training takes around 30 hours on a single RTX 3090 GPU.

For DynACL++, we need to first generate pseudo labels:

    python assign_label.py --checkpoint PATH_TO_PRETRAINED_MODEL --dataset DATASET_NAME --data PATH_TO_DATASET

Pseudo labels will be stored in the same folder as the checkpoint.

To perform DynACL++ finetuning, we provide two scripts. Both scripts will first perform DynACL++ training, then perform evaluation on these trained models.

If you want to train the model for linear evaluation (SLF & ALF), run

    python LPAFT.py --experiment EXP_PATH --label_path PATH_TO_LABELS --checkpoint PATH_TO_PRETRAINED_MODEL --dataset DATASET_NAME --data PATH_TO_DATASET

The output model ``model.pt`` only contains the finetuned adversarial route.  This script also includes the evaluation for SLF and ALF settings, you may specify your desired evaluation settings by ``--evaluation_mode``. 

If you want to train the model for adversarial full evaluation (AFF) or semi-supervised evaluation, run 

    python LPAFT_AFF.py --experiment EXP_PATH --label_path PATH_TO_LABELS --checkpoint PATH_TO_PRETRAINED_MODEL --dataset DATASET_NAME --data PATH_TO_DATASET

The output model ``model.pt`` contains both route of the backbone. This result will also automatcally perform adversarial adversarial full finetuning (AFF) after training ends.

All training results (checkpoints and loggers) will be stored in ``./experiments/EXP_PATH``. Standard accuracy (SA) and robust accuracy results are stored in ``log.txt``, while the auto-attack (AA) results are stored in ``robustness_results.txt``.

## Evaluation

### DynACL (SLF & ALF) 

    python test_LF.py --experiment EXP_PATH --checkpoint PATH_TO_PRETRAINED_MODEL --dataset DATASET_NAME --data PATH_TO_DATASET --cvt_state_dict --bnNameCnt 1 --evaluation_mode EVALUATION_MODE

### DynACL (AFF) 

    python test_AFF.py --experiment EXP_PATH --checkpoint PATH_TO_PRETRAINED_MODEL --dataset DATASET_NAME --data PATH_TO_DATASET

### DynACL++ (SLF & ALF & AFF)
For DynACL++ evalution, the above training scripts (``LPAFT.py`` and ``LPAFT_AFF.py``) already contain the evaluation part. You may also use ``test_LF.py`` and ``test_AFF.py`` for evaluation. Be aware that ``LPAFT.py`` will give you a single-BN checkpoint, so there's no need to specify ``cvt_state_dict``.

### DynACL++ (semi-supervised)
We borrow the semi-supervised evaluation code from [ACL](https://github.com/VITA-Group/Adversarial-Contrastive-Learning). You may clone their code and follow their semi-supervised evaluation steps. Note that you need to use ``LPAFT_AFF.py`` to train a model with dual BN.

## Pretrained weights
---

Pretrained after DynACL

Finetuned after DynACL++ (single-BN)

Finetuned after DynACL++ (dual-BN)

## Citation
---
If you find our paper inspiring, please cite

    @inproceedings{DynACL,
        title = {Rethinking the Effect of Data Augmentation in Adversarial Contrastive Learning},
        author = {Luo, Rundong and Wang, Yifei and Wang, Yisen},
        booktitle = {ICLR},
        year = {2023},
    }

## Acknowledgements

Some of our code are borrowed from [ACL](https://github.com/VITA-Group/Adversarial-Contrastive-Learning). Thanks for their great work!



