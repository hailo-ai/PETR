import argparse
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
from pathlib import Path
from tqdm import tqdm

from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmcv.runner.base_module import BaseModule
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector

# sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
sys.path.insert(0, os.getcwd())
from projects.mmdet3d_plugin.models.detectors.petr3d import Petr3D


class CalibGenerator(Petr3D):
    def __init__(self, cfg_dict=None, calib_set_size=1024, device='cuda:0'):
        super(CalibGenerator, self).__init__(**cfg_dict)
        self.calib_set_size = calib_set_size
        self.device = device

    def forward(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
        return self.simple_test(img_metas[0], img[0], **kwargs)
    
    def simple_test(self, img_metas, img=None, **kwargs):
        """Test function without augmentaiton."""
        img = img.to(self.device)
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        x = img_feats[self.pts_bbox_head.position_level]
        batch_size, num_cams,_ , H, W = x.size()

        input_img_h, input_img_w, _ = img_metas[0]['pad_shape'][0]
        masks = x.new_ones(
            (batch_size, num_cams, input_img_h, input_img_w))
        for img_id in range(batch_size):
            for cam_id in range(num_cams):
                img_h, img_w, _ = img_metas[img_id]['img_shape'][cam_id]
                masks[img_id, cam_id, :img_h, :img_w] = 0
        masks = F.interpolate(
            masks, size=(H, W)).to(torch.bool)

        if self.pts_bbox_head.with_position:
            coords_position_embeding, _ = self.pts_bbox_head.position_embeding(img_feats, img_metas, masks)
        
        mlvl_feats = x.flatten(0,1).permute(1,0,2,3).reshape(batch_size, self.pts_bbox_head.in_channels, num_cams, -1)
        return mlvl_feats.permute(0, 2, 3, 1), coords_position_embeding.permute(0, 2, 3, 1) # bs,ch,h,w --> bs,h,w,ch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='model config path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--img-dims', nargs=2, type=int, default=None, metavar=('height','width'), help='Change input resolution (height, width). Overrides cfg values')
    parser.add_argument('--num-images', type=int, default=int(1e6), help='How many images to iterate over for data generation')
    parser.add_argument('--save-dir', type=str, default=None, help='Folder to save calibration sets')
    parser.add_argument('--net-name', type=str, default=None, help='Model name')
    parser.add_argument('--gen-calib-set', action='store_true',
                        help='Generate NPZ calibration set for transformer holding backbone feature maps and 3d coordinates positional embedding data')
    args = parser.parse_args()
    return args


def main(device='cuda:0'):
    args = parse_args()
    num_images = args.num_images
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True

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

    if args.img_dims is not None:
        print(f'Changing input resolution from'
              f' {cfg.ida_aug_conf.final_dim} to {tuple(args.img_dims)}')
        cfg.data.test.pipeline[2]['data_aug_conf']['final_dim'] = args.img_dims
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval()

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    from projects.mmdet3d_plugin.models.backbones.repvgg import RepVGG, repvgg_model_convert
    cfg.model.pop('type')
    calib_model = CalibGenerator(cfg.model, device=device)
    calib_model.load_state_dict(model.state_dict())   
    if isinstance(calib_model.img_backbone, RepVGG):
        img_backbone_deploy = repvgg_model_convert(model.img_backbone)
        calib_model.img_backbone = img_backbone_deploy
        calib_model.img_backbone.deploy = True

    calib_model.eval()
    calib_model.to(device)

    gen_calib_set = args.gen_calib_set
    if gen_calib_set:
        calib_set = {'backbone': [],
                     'transformer':
                        {}
                    }
    count = 0
    np.random.seed(0)

    work_dir = Path(os.getcwd()) / 'coords3d_positional_embedding' if args.save_dir is None else args.save_dir
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    for i, data in tqdm(enumerate(data_loader), total=min(num_images, len(data_loader))):
        img_metas = data['img_metas'][0].data[0]
        img = data['img'][0].data[0].to(device)
        bs, num_cams = img.size(0), img.size(1)

        img_npy = img[0,0].cpu().numpy().transpose(1,2,0)
        shape = img_npy.shape
        calib_data = img_npy.reshape(1, *shape)

        bb_stride = 16
        if hasattr(cfg.model.pts_bbox_head, 'position_level'):
            assert cfg.model.pts_bbox_head.position_level in [0, 1]
            bb_stride = 16 if cfg.model.pts_bbox_head.position_level==0 else 32
        # tokens = int((shape[0]/bb_stride) * (shape[1]/bb_stride) * num_cams)
        # embed_dims = cfg.model.pts_bbox_head.transformer.decoder.transformerlayers.attn_cfgs[0].embed_dims

        if gen_calib_set and count==0:
            calib_set['backbone'] = np.zeros((num_images, *shape))
            calib_set['transformer']['mlvl_feats'] = np.zeros((num_images, 
                                                               num_cams,
                                                               int(shape[0]/bb_stride * shape[1]/bb_stride),
                                                               calib_model.pts_bbox_head.in_channels,
                                                               ))
            calib_set['transformer']['coords_3d_pe'] = np.zeros((num_images, 
                                                                 num_cams,
                                                                 int(shape[0]/bb_stride * shape[1]/bb_stride),
                                                                 calib_model.pts_bbox_head.embed_dims,
                                                                 ))

        mlvl_feats, coords_3d_pe = calib_model(img_metas=data['img_metas'][0].data, img=data['img'][0].data)
        mlvl_feats = mlvl_feats.detach().cpu().numpy()
        coords_3d_pe = coords_3d_pe.detach().cpu().numpy()

        if gen_calib_set and count < num_images:
            calib_set['backbone'][count] = calib_data
            calib_set['transformer']['mlvl_feats'][count] = mlvl_feats
            calib_set['transformer']['coords_3d_pe'][count] = coords_3d_pe

        transformer_inputs = {}
        transformer_inputs['input_layer1'] = mlvl_feats
        transformer_inputs['input_layer2'] = coords_3d_pe

        coords3d_path = work_dir / f'coords3d_pe_{count}.npy'
        np.save(coords3d_path, coords_3d_pe.squeeze(0))

        count += 1
        if count == num_images:
            break

    print(f'\n 3D coordiantes positional embeddings data saved at {work_dir}')

    # Save transformer calibration set
    if gen_calib_set:
        suf = Path(args.config).suffix
        net_name = Path(args.config).name.split(suf)[0]if args.net_name is None else args.net_name
        calib_set_transformer = {
            net_name + '/input_layer1': calib_set['transformer']['mlvl_feats'],
            net_name + '/input_layer2': calib_set['transformer']['coords_3d_pe']
        }
        calib_transformer_path = work_dir.parent / f'{net_name}_calib_set_transformer_{num_images}.npz'
        np.savez(calib_transformer_path, **calib_set_transformer)

        print(f"Calibration set with {count} images for transformer saved at: {calib_transformer_path}\n")
    return


if __name__ == "__main__":
    main()

"""
This script needs to be run from the PETR/ folder, e.g.

*** Generate 3D positional embedding data
cd /workspace/PETR
CUDA_VISIBLE_DEVICES=0 python3 tools/gen_calib_set.py
    projects/configs/petrv2/petrv2_fcos3d_repvgg_b0x32_BN_q_304_decoder_3_UN_800x320.py 
    <train.pth>
    --save-dir <path_to_save_dir>

*** Also generate calibration set for transformer, holding backbone feature maps and 3D positional embedding
cd /workspace/PETR
CUDA_VISIBLE_DEVICES=0 python3 tools/gen_calib_set.py
    projects/configs/petrv2/petrv2_fcos3d_repvgg_b0x32_BN_q_304_decoder_3_UN_800x320.py 
    <train.pth>
    --num-images 64
    --save-dir <path_to_save_dir>
    --net-name petrv2_repvggB0_transformer_pp_800x320

"""