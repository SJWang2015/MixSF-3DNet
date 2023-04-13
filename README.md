# MixSF-3DNet
This code is for the work: Mixer-based Symmetric Scene Flow Estimation from 3D Point Clouds (MixSF-3DNet).

<img src="https://github.com/SJWang2015/MixSF-3DNet/blob/main/media/poster-v3.png" width="60%">

### Abstract

The scene flow estimation aims at achieving high-quality point-wise motion fields through accurate point-pair correspondence maps and robust scene flow features. The major challenges of 3D point clouds based scene flow (SF) estimation include mis-registration of point clouds, object occlusions, and non-uniform scene flow upsampling in the multi-scale SF estimation framework. This paper presents a scene flow estimation framework, which consists of an MLP-Mixer operation based correspondence-weighted cost volume (CV) module, a symmetric cost volume approach for scene flow estimation, and a geometric/semantic feature based upsampling strategy. The novelty of this work is threefold: (1) developing a progressive mutual improvement framework for scene flow estimation through the integration of the same operator of the cost volume module and scene flow estimator into one module; (2) developing a symmetric inter-frame correlation feature extraction method through CV estimation using MLP-Mixer operations; and (3) developing an upsampling strategy based on both the semantic and geometric point feature similarities between sparse and dense samples. The proposed method is trained and verified using the FlyingThings3D and Kitti datasets, respectively, with/without occluded scene flows. The experiment results demonstrate the superior performance of the proposed method over the state-of-the-art baseline methods.

#### Training and Evaluation

To train the model, simply execute the shell script `command_train.sh`. Batch size, learning rate etc are adjustable. The model used for training is `model_concat_upsa.py`.

```
sh cmd.sh
```

To evaluate the model, simply execute the shell script `command_evaluate_flyingthings.sh`.

```
sh command_eval_ft3d.sh
```


### License
Our code is released under MIT License (see LICENSE file for details).

