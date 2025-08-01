## Common Issues

### 1.) Known Issue with gym 0.21.0

If you encounter issues installing `gym-retro` with `gym` version 0.21.0, you may come across a bug related to TensorFlow dependencies. The issue arises due to conflicting dependencies between `gym` and TensorFlow.

However, this issue can be resolved by following the workaround provided [here](https://github.com/openai/gym/issues/3202).

"pip install git+https://github.com/openai/gym.git@9180d12e1b66e7e2a1a622614f787a6ec147ac40"

installs gym==0.21 with the typo opencv-python>=3. fixed.

Please ensure to follow the steps mentioned in the linked GitHub issue to successfully install `gym-retro` with `gym` version 0.21.0.

### 2.) Issue with Tensorflow and Apple M1 chip

The issue with TensorFlow and Apple M1 chip primarily arises from the architecture difference between traditional x86_64 CPUs and the ARM-based architecture used in Apple's M1 chips.

#### Compatibility
- TensorFlow initially lacked native support for the ARM architecture used in Apple's M1 chip, leading to installation and runtime issues for M1-powered Mac users.
- Compatibility problems extended to dependencies like NumPy, causing conflicts during installation alongside TensorFlow on M1-powered Macs.

#### Performance
- Even after making TensorFlow compatible with ARM architecture, performance on M1 chips was suboptimal compared to Intel-based Macs, attributed to optimization and hardware architecture differences.
- Apple's Rosetta translation technology introduced performance overhead, further impacting TensorFlow performance on M1 chips.

Solution(For python 3.8.5): [Link](https://stackoverflow.com/questions/65383338/zsh-illegal-hardware-instruction-python-when-installing-tensorflow-on-macbook)

To address compatibility issues with TensorFlow and Apple M1 chips, you can install the necessary wheel files separately. The solution provided above includes a link to download the wheel files [Link](https://drive.google.com/drive/folders/1oSipZLnoeQB0Awz8U68KYeCPsULy_dQ7).
