# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
from .vovnet import VoVNet
from .vovnetcp import VoVNetCP
from .repvgg import RepVGG
__all__ = ['VoVNet', 'VoVNetCP', 'RepVGG']

