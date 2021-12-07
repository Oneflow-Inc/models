Modified from https://github.com/rosinality/stylegan2-pytorch

`Q`: What is FusedLeakyReLU(x)

`A`: It's equal to ```scale*LeakyReLU(x+bias)```

`Q`: What is upfirdn2d(x)

`A`: Adaptive Discriminator Augmentation(ADA), as mentioned in paper Training Generative Adversarial Networks with Limited Data 