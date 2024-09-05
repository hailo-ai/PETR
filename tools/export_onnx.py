import argparse
import sys
from pathlib import Path
import torch
import os
import numpy as np
import onnx
from onnxsim import simplify
from onnx.compose import merge_models
import json
import torch.nn.functional as F

from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector


sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
from projects.mmdet3d_plugin.models.detectors.petr3d import Petr3D
from projects.mmdet3d_plugin.models.backbones.repvgg import RepVGG, repvgg_model_convert
from projects.mmdet3d_plugin.models.dense_heads.petrv2_head import inverse_sigmoid


class Petr3D_Split(Petr3D):
    def __init__(self, split=None, cfg_dict=None, petr_version='v2'):
        super(Petr3D_Split, self).__init__(**cfg_dict)
        self.onnx_split_choices = ['backbone', 'transformer']
        assert split in self.onnx_split_choices, f"{split} is not one of {self.onnx_split_choices}"
        self.split = split
        self.petr_version_choices = ['v1', 'v2']
        assert petr_version in self.petr_version_choices, f"{petr_version} is not one of {self.petr_version_choices}"
        self.petr_version = petr_version
    
    def forward(self, x, img_metas):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        if self.split == 'backbone':
            return self.foward_backbone(x, img_metas)
        elif self.split == 'transformer':
            return  self.foward_transformer(x, img_metas)
        else:
            raise ValueError(f"Unsupported split type {self.split}. Valid values are {self.onnx_split_choices}")

    def foward_backbone(self, img, img_metas):
        img = [img] if img is None else img
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        if self.petr_version == 'v1':
            # For PETR-v1 features projection is included in backbone
            x = img_feats[self.pts_bbox_head.position_level]
            x = self.pts_bbox_head.input_proj(x.flatten(0,1))
            return x
        elif self.petr_version == 'v2':
            return img_feats
        else:
            raise ValueError(f"Unsupported {self.petr_version} PETR version. Valid values are {self.petr_version_choices}")

    def foward_transformer(self, img_feats, img_metas):
        outs = self.pts_bbox_head(img_feats, img_metas)
        return outs

class MaskModel(torch.nn.Module):
    def __init__(self, num_query, num_heads, num_tokens, num_splits):
        super(MaskModel, self).__init__()
        self.num_query = num_query
        self.num_heads = num_heads
        self.num_tokens = num_tokens
        self.num_splits = num_splits

    def forward(self, x):
        x = torch.zeros((self.num_query, self.num_heads//self.num_splits, self.num_tokens)).permute(1, 0, 2)
        return x


class SinEmbed(torch.nn.Module):
    def __init__(self, sin_embed_path=None):
        super(SinEmbed, self).__init__()
        self.sin_embed_path = sin_embed_path

    def forward(self, x):
        sin_embed = torch.load(self.sin_embed_path)
        return sin_embed


def reshape_onnx(onnx_path, cfg_path, sin_embed_path=None):
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    
    MLVL_FEATS = cfg['MLVL_FEATS_NAMES']
    COORDS_PE = cfg['COORDS_PE_NAMES']
    SIN_EMBED_NAMES = cfg['SIN_EMBED_NAMES']
    MASK_NAMES = cfg['MASK_NAMES']
    OUTPUT_NAMES = cfg['OUTPUT_NAMES']

    onnx_model = onnx.load_model(onnx_path)
    extractor = onnx.utils.Extractor(onnx_model)

    onnx_model = extractor.extract_model(
        input_names = MLVL_FEATS + COORDS_PE + SIN_EMBED_NAMES + MASK_NAMES,
        output_names = OUTPUT_NAMES,
    )

    NUM_TOKENS = cfg['NUM_TOKENS']
    NUM_QUERY = cfg['NUM_QUERY']
    NUM_HEADS = cfg['NUM_HEADS']
    NUM_SPLITS = cfg['NUM_SPLITS']

    mask = MaskModel(NUM_QUERY, NUM_HEADS, NUM_TOKENS, NUM_SPLITS)
    sin_embed = SinEmbed(sin_embed_path=sin_embed_path)
    dummy_input = torch.randn(1, (NUM_HEADS//NUM_SPLITS)*NUM_TOKENS, 1, NUM_QUERY)

    torch.onnx.export(mask, dummy_input, 'mask_input.onnx', opset_version=13)
    torch.onnx.export(sin_embed, dummy_input, 'sin_embed_input.onnx', opset_version=13)

    mask_input = onnx.load_model('mask_input.onnx')
    sin_embed_input = onnx.load_model('sin_embed_input.onnx')

    mask_input.ir_version = onnx_model.ir_version
    sin_embed_input.ir_version = onnx_model.ir_version

    onnx_model = merge_models(sin_embed_input,
                              onnx_model,
                              io_map=[(sin_embed_input.graph.output[0].name, SIN_EMBED_NAMES[0])],
                              prefix1="sin_embed",
    )
    for i in range(len(MASK_NAMES)):
        onnx_model = merge_models(mask_input,
                                  onnx_model,
                                  io_map=[(mask_input.graph.output[0].name, MASK_NAMES[i])],
                                  prefix1='mask',
        )

    return onnx_model


def get_sin_embed(model, img, img_metas):
    position_level = model.pts_bbox_head.position_level
    batch_size, num_cams, ch, img_feats_h, img_feats_w = img[position_level].shape
    input_img_h, input_img_w, _ = img_metas[0][0]['pad_shape'][0]
    masks = torch.ones((batch_size, num_cams, input_img_h, input_img_w))
    img_h, img_w, _  = img_metas[0][0]['img_shape'][0]

    for img_id in range(batch_size):
        for cam_id in range(num_cams):
            img_h, img_w, _ = img_metas[0][img_id]['img_shape'][cam_id]
            masks[img_id, cam_id, :img_h, :img_w] = 0
    masks = F.interpolate(masks, size=(img_feats_h, img_feats_w)).to(torch.bool)
    sin_embed = model.pts_bbox_head.positional_encoding(masks)
    ch = sin_embed.size(2)
    sin_embed = model.pts_bbox_head.adapt_pos3d(sin_embed.flatten(0,1).permute(1,0,2,3).reshape(1, ch, num_cams,-1))

    return sin_embed.detach().cpu()


def get_reference_points(model, img):
    reference_points = model.pts_bbox_head.reference_points.weight
    batch_size = img[0].shape[0] if isinstance(img, list) else img.shape[0]
    reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
    reference = inverse_sigmoid(reference_points)
    return reference.detach().numpy()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='model config path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('-o', '--opset', type=int, default=13)
    parser.add_argument('--ncams', type=int, default=6, help='Number of cameras')
    parser.add_argument('--img-dims', nargs=2, type=int, default=None, metavar=('height','width'),
                        help='Change input resolution (height, width). Overrides cfg values')
    parser.add_argument('--mha-groups', type=int, default=1, help='How many groups to split the multi attention heads into')
    parser.add_argument('--split', default='transformer', type=str, help="Split and export backbone / transformer part of model",
                        choices=['backbone', 'transformer'], required=True)
    parser.add_argument('--petr-version', default='v2', type=str, help="PETR version", choices=['v1', 'v2'])
    parser.add_argument('--reshape-cfg', default=None, type=str, help="ONNX reshape config path")
    parser.add_argument('--out', default='petr_v2.onnx', type=str, help="Name for the onnx output")
    parser.add_argument('--no-onnx-reshape', action='store_true',
                        help="Disable auto merging of middle and post processing into ONNX. " 
                             "This will require further ONNX processing with `tools/onnx_reshape.py` script.")
    args = parser.parse_args()
    return args


def main(device='cpu'):
    args = parse_args()

    if args.ncams > 6:
        raise ValueError(f"Maximum number of cameras is 6 but got {args.ncam}")

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                plg_lib = importlib.import_module(_module_path)

    # build the model and load checkpoint
    if args.img_dims is not None:
        print(f'Changing input resolution from'
              f' {cfg.ida_aug_conf.final_dim} to {tuple(args.img_dims)}')
        ind = [i for (i,x) in enumerate(cfg.data.test.pipeline)
                if x['type']=='ResizeCropFlipImage'][0]
        cfg.data.test.pipeline[ind]['data_aug_conf']['final_dim'] = args.img_dims
    cfg.model.pts_bbox_head.transformer.decoder.\
        transformerlayers.attn_cfgs[1].num_head_split = args.mha_groups
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval()

    # get example data from dataset
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)
    for i, data in enumerate(data_loader):
        img_metas = data['img_metas'][0].data
        img = data['img'][0].data[0].to(device)
        if args.split == 'transformer':
            B, N, C, H, W = img.shape
            out_stride = 16
            C = cfg.model.pts_bbox_head.in_channels
            if args.petr_version == 'v1':
                N = 1 * args.ncams
            elif args.petr_version == 'v2':
                N = 2 * args.ncams
            else:
                raise ValueError(f"Unrecognized PETR version {args.petr_version}")
            img = [torch.zeros(B, N, C, H//out_stride, W//out_stride),
                   torch.zeros(B, N, C, H//(2*out_stride), W//(2*out_stride))]
    
            if img[0].shape[1] < 12:
                for k, v in img_metas[0][0].items():
                    if isinstance(v, list):
                        img_metas[0][0][k] = v[:img[0].shape[1]]
        break
    # convert img metas to torch tensors
    keys_to_remove = []
    for i in range(len(img_metas)):
        for j in range(len(img_metas[i])):
            for key, value in img_metas[i][j].items():
                if isinstance(value, np.ndarray):
                    img_metas[i][j][key] = torch.from_numpy(value)
                elif isinstance(value, list):
                    for k in range(len(value)):
                        if isinstance(value[k], np.ndarray):
                            img_metas[i][j][key][k] = torch.from_numpy(value[k])
                        elif isinstance(value, tuple):
                            img_metas[i][j][key][k] = torch.tensor(value[k])
                elif isinstance(value, dict):
                    for kk, vv in value.items():
                        if isinstance(vv, np.ndarray):
                            img_metas[i][j][key][kk] = torch.from_numpy(vv)
                elif isinstance(value, type):
                    keys_to_remove.append(key)
    for remove in keys_to_remove:
        img_metas[0][0].pop(remove)
    img_metas[0][0]['box_mode_3d'] = int(img_metas[0][0]['box_mode_3d'])
    
    # split backbone / transformer part of model
    cfg.model.pop('type')
    split_model = Petr3D_Split(args.split, cfg.model, args.petr_version)
    split_model.load_state_dict(model.state_dict())    
    if isinstance(split_model.img_backbone, RepVGG):
        img_backbone_deploy = repvgg_model_convert(model.img_backbone)
        split_model.img_backbone = img_backbone_deploy
        split_model.img_backbone.deploy = True
    split_model.eval()

    # export ONNX
    with torch.no_grad():
        torch.onnx.export(split_model,
                          (img, img_metas[0]), args.out,
                          do_constant_folding=False,
                          opset_version=args.opset)
        model_onnx = onnx.load(args.out)
        print('Simplifying model..')
        model_simp, check = simplify(model_onnx)
        onnx.save(model_simp, args.out)
    if (args.split == 'backbone' or args.no_onnx_reshape):
        print(f"Simplified model saved at: {args.out}")

    elif (args.split == 'transformer' and not args.no_onnx_reshape):
        # Auto ONNX reshape adding middle and post process to ONNX
        err_msg = None
        if not args.reshape_cfg:
            err_msg = f"Reshape json config not provided. Please use --reshape-cfg to " \
                       "set it path or use --no-onnx-reshape to manually reshape the resulting ONNX with tools/onnx_reshape.py"
            raise ValueError(err_msg)
        elif args.petr_version == 'v1':
            err_msg = f"Auto onnx reshape is not supported for PETR-v1. Please use --no-onnx-reshape and " \
                       "manually reshape the resulting ONNX with tools/onnx_reshape.py"
        if err_msg:
            raise ValueError(err_msg)

        sin_embed = get_sin_embed(split_model, img, img_metas)
        sin_embed_path = 'sin_embed.pt'
        torch.save(sin_embed, sin_embed_path)

        print('Reshaping ONNX model..')
        model_onnx = reshape_onnx(onnx_path=args.out,
                                  cfg_path=args.reshape_cfg,
                                  sin_embed_path=sin_embed_path)
        print('Simplifying model..')
        model_simp, check = simplify(model_onnx)
        onnx.save_model(model_simp, args.out)
        print(f"Reshaped simplified model saved at {args.out}")

        # save reference points for post processing
        reference = get_reference_points(split_model, img)
        ref_path = 'reference_points.npy'
        np.save(ref_path, reference)
        print(f"Reference points for postprocessing saved at {ref_path}")

if __name__ == '__main__':
    main()


"""
Example usage:
-------------

(*) Backbone export:

    python tools/export_onnx.py projects/configs/petrv2/petrv2_fcos3d_repvgg_b0x32_BN_q_304_decoder_3_UN_800x320.py
        </path/to/pth>
        --split backbone
        --out petrv2_backbone.onnx

(*) Transformer export:

    python tools/export_onnx.py projects/configs/petrv2/petrv2_fcos3d_repvgg_b0x32_BN_q_304_decoder_3_UN_800x320.py
        </path/to/pth>
        --split transformer 
        --out petrv2_transformer.onnx
        --reshape-cfg tools/onnx_reshape_cfg_repvgg_b0x32_BN2D_decoder_3_q_304_UN_800x320.json
"""