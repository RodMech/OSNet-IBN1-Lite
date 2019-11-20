# OSNet-IBN (width x 1.0) Lite



## Contents

This project is a reduced version of [OSNet](Omni-Scale Feature Learning for Person Re-Identification), a network designed by Kaiyang Zhou. It has been extracted from [Torxreid](https://github.com/KaiyangZhou/deep-person-reid), the main framework for training and testing reID CNNs. 

Many features from the original backend have been isolated. Torxvision dependency has been deleted to keep docker container as light as possible. The required methods have been extracted from the source code. 

The net is able to work both in GPU and CPU. 



## Requirements

The required packages have been version-pinned in the *requirements.txt*.

`torch==1.2`
`numpy==1.16.4`
`Pillow`
`pandas`



## Dockerfile

This version includes two Dockerfiles: 

- `dev-nogpu.dockerfile`
- `dev-gpu.dockerfile`: CUDA 10 // CUDNN 7

For testing the GPU container, the user could use the following command: 

`docker run --runtime=nvidia  -it --volume /home/ec2-user/reid:/opt/project reid-dev:gpu /bin/bash`

Or a docker-compose file could be used. 



## Weights

Default weights are provided by the author at [Torxreid model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO). 



## Performance

It has been tested on a EC2 instance p2.xlarge, with an average of 35 images per second.  



## Citation


    @article{torchreid,
      title={Torchreid: A Library for Deep Learning Person Re-Identification in Pytorch},
      author={Zhou, Kaiyang and Xiang, Tao},
      journal={arXiv preprint arXiv:1910.10093},
      year={2019}
    }
    
    @inproceedings{zhou2019osnet,
      title={Omni-Scale Feature Learning for Person Re-Identification},
      author={Zhou, Kaiyang and Yang, Yongxin and Cavallaro, Andrea and Xiang, Tao},
      booktitle={ICCV},
      year={2019}
    }
    
    @article{zhou2019learning,
      title={Learning Generalisable Omni-Scale Representations for Person Re-Identification},
      author={Zhou, Kaiyang and Yang, Yongxin and Cavallaro, Andrea and Xiang, Tao},
      journal={arXiv preprint arXiv:1910.06827},
      year={2019}
    }