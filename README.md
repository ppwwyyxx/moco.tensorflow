
Implement and reproduce results of the following papers:

* [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722)
* [Improved Baselines with Momentum Contrastive Learning](https://arxiv.org/abs/2003.04297)

## Dependencies:

* TensorFlow 1.14 or 1.15, built with XLA support
* [Tensorpack](https://github.com/tensorpack/tensorpack/) ≥ 0.10
* Horovod ≥ 0.19 built with Gloo & NCCL support
* TensorFlow [zmq_ops](https://github.com/tensorpack/zmq_ops)
* OpenCV
* the `taskset` command (from the `util-linux` package)

## Unsupervised Training:

To run MoCo pre-training on a machine with 8 GPUs, use:
```
horovodrun -np 8 --output-filename moco.log python main_moco.py --data /path/to/imagenet
```

Add `--v2` to train MoCov2,
which uses an extra MLP layer, extra augmentations, and cosine LR schedule.


## Linear Classification:
To train a linear classifier using the pre-trained features, run:
```
./main_lincls.py --load /path/to/pretrained/checkpoint --data /path/to/imagenet
```

## Results
Training was done in a machine with 8 V100s, >200GB RAM and 80 CPUs.

200 epochs of pre-training takes ~53h;
100 epochs of linear classifier training takes ~8h.

Pre-trained models (without linear classifier):
* MoCo v1: [:arrow_down:](https://github.com/ppwwyyxx/moco.tensorflow/releases/download/v/MoCo_v1.npz).
	(60.9% linear classification accuracy)
* MoCo v2: [:arrow_down:](https://github.com/ppwwyyxx/moco.tensorflow/releases/download/v/MoCo_v2.npz).
	(66.7% linear classification accuracy)

## Notes:

* Horovod with Gloo is recommended. Horovod with MPI is not tested and may crash due to how we use forking.
* If using TensorFlow without XLA support, you can modify `main_*.py` to replace `xla.compile` by a naive forward.
* Official PyTorch code is at [facebookresearch/moco](https://github.com/facebookresearch/moco).
