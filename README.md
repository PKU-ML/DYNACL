# [ICLR23] Rethinking the Effect of Data Augmentation in Adversarial Contrastive Learning

Authors: Rundong Luo*, Yifei Wang*, Yisen Wang from Peking University


## Introduction
Recent works have shown that self-supervised learning can achieve remarkable robustness when integrated with adversarial training (AT). However, the robustness gap between supervised AT (sup-AT) and self-supervised AT (self-AT) remains significant. Motivated by this observation, we revisit existing self-AT methods and discover an inherent dilemma that affects self-AT robustness: either strong or weak data augmentations are harmful to self-AT, and a medium strength is insufficient to bridge the gap. To resolve this dilemma, we propose a simple remedy named DynACL (Dynamic Adversarial Contrastive Learning). In particular, we propose an augmentation schedule that gradually anneals from a strong augmentation to a weak one to benefit from both extreme cases. Besides, we adopt a fast post-processing stage for adapting it to downstream tasks. Through extensive experiments, we show that DynACL can improve the state-of-the-art self-AT robustness by 8.84% under Auto-Attack on the CIFAR-10 dataset, and can even outperform vanilla supervised adversarial training. We demonstrate that self-supervised AT can attain even better robustness than supervised AT for the first time.

## Environment
A standard pytorch environment (>=1.0.0) with basic packages (e.g., numpy, pickle) is enough. Besides, to evaluate under the Auto-Attack benchmark, the [autoattack](https://github.com/fra31/auto-attack) package is required. Run the following code to install:

    pip install git+https://github.com/fra31/auto-attack

Additionally, one package is required for generating pseudo labels. We recommend the [kmeans-pytorch](https://github.com/subhadarship/kmeans_pytorch) package (can be directly installed by pip). An alternative is the sklearn.

## Data
CIFAR10, CIFAR100, and STL10 dataset are required. You may manually download them and put in the ``./data`` folder, or directly run our provided scripts to automatically download these datasets.

## Training

We assume you have a GPU with no less than 16GB memory (e.g., 3090). Evaluation require fewer memory (less than 8GB).

For DynACL, run

    python train_DynACL.py --experiment EXP_PATH --dataset DATASET_NAME --data PATH_TO_DATASET

All training results (checkpoints and loggers) will be stored in ``./experiments/EXP_PATH``. Training takes around one day on a single RTX 3090 GPU.

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

Pretrained after [DynACL](https://disk.pku.edu.cn:443/link/6753909DC4EC5E17AD4A5290A55EA6F0)

Finetuned after [DynACL++(single-BN)](https://disk.pku.edu.cn:443/link/575295C2C24F6F9404C50765D5899FD8)

Finetuned after [DynACL++(dual-BN)](https://disk.pku.edu.cn:443/link/6543A73CECDF6F88EDD10FA7EB95508E)

## Citation
If you find our paper inspiring, please cite

    @inproceedings{DynACL,
        title = {Rethinking the Effect of Data Augmentation in Adversarial Contrastive Learning},
        author = {Luo, Rundong and Wang, Yifei and Wang, Yisen},
        booktitle = {ICLR},
        year = {2023},
    }

## Acknowledgements

Some of our code are borrowed from [ACL](https://github.com/VITA-Group/Adversarial-Contrastive-Learning). Thanks for their great work!



