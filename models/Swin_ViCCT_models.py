from timm.models.swin_transformer import _create_swin_transformer
import torch
import torch.nn as nn
from timm.models import register_model


# ============================================================================================ #
#                                     HEAD TO PERFORM REGRESSION                               #
# ============================================================================================ #

class ViCCTRegressionHead(nn.Module):
    def __init__(self, crop_size, embed_dim):
        super().__init__()

        self.regression_head = nn.ModuleDict({
            'lin_scaler': nn.Sequential(
                nn.Linear(embed_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256)
            ),
            'folder': nn.Fold((crop_size, crop_size), kernel_size=16, stride=16)
        })

    def forward(self, pre_den):
        # Modern timm's Swin backbone returns features as (B, H, W, C) (channels-last).
        # nn.Linear only cares about the last dim, so that's fine, but nn.Fold needs a
        # 3D (B, C, L) tensor, so we flatten the spatial grid into a single "patch" axis first.
        if pre_den.ndim == 4:
            B, H, W, C = pre_den.shape
            pre_den = pre_den.reshape(B, H * W, C)  # (B, L, C)

        pre_den = self.regression_head['lin_scaler'](pre_den)  # (B, L, 256)
        pre_den = pre_den.transpose(1, 2)                      # (B, 256, L)
        den = self.regression_head['folder'](pre_den)          # (B, 1, crop_size, crop_size)

        return den


class DistilledRegressionTransformer(nn.Module):
    def __init__(self, base_model, **kwargs):
        super().__init__()

        # Keep only patch_embed + the transformer stages; drop the final norm + classification head.
        self.base_model = nn.Sequential(*list(base_model.children())[:-2])
        self.regression_head = ViCCTRegressionHead(224, kwargs['embed_dim'] * 4)

    def forward(self, x):
        x = self.base_model(x)
        den = self.regression_head(x)
        return den


# ============================================================================================ #
#                                           THE MODELS                                         #
# ============================================================================================ #

def _clean_kwargs(kwargs):
    """ timm's `create_model` gets called with drop_rate/drop_path_rate/drop_block_rate=None
    by the notebook. Modern timm expects floats (or the kwarg to simply be absent), so we
    strip out anything set to None instead of forwarding it blindly. """
    return {k: v for k, v in kwargs.items() if v is not None}


@register_model
def Swin_ViCCT_small(init_path=None, pretrained_cc=False, **kwargs):
    """ Swin-S @ 224x224, trained ImageNet-1k """

    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 18), num_heads=(3, 6, 12),
        **_clean_kwargs(kwargs))

    base_model = _create_swin_transformer('swin_small_patch4_window7_224', pretrained=False, **model_kwargs)

    if init_path and not pretrained_cc:
        base_model = init_model_state(base_model, init_path)

    full_model = DistilledRegressionTransformer(base_model, **model_kwargs)

    if init_path and pretrained_cc:
        full_model = load_pretrained(full_model, init_path)

    full_model.crop_size = 224

    return full_model


@register_model
def Swin_ViCCT_base(init_path=None, pretrained_cc=False, **kwargs):
    """ Swin-B @ 224x224, pretrained ImageNet-1k """

    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18), num_heads=(4, 8, 16),
        **_clean_kwargs(kwargs))

    base_model = _create_swin_transformer('swin_base_patch4_window7_224', pretrained=False, **model_kwargs)

    if init_path and not pretrained_cc:
        base_model = init_model_state(base_model, init_path)

    full_model = DistilledRegressionTransformer(base_model, **model_kwargs)

    if init_path and pretrained_cc:
        full_model = load_pretrained(full_model, init_path)

    full_model.crop_size = 224

    return full_model


@register_model
def Swin_ViCCT_large(init_path=None, pretrained_cc=False, **kwargs):
    """ Swin-L @ 224x224, trained ImageNet-1k """

    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=192, depths=(2, 2, 18), num_heads=(6, 12, 24),
        **_clean_kwargs(kwargs))

    base_model = _create_swin_transformer('swin_large_patch4_window7_224', pretrained=False, **model_kwargs)

    if init_path and not pretrained_cc:
        base_model = init_model_state(base_model, init_path)

    full_model = DistilledRegressionTransformer(base_model, **model_kwargs)

    if init_path and pretrained_cc:
        full_model = load_pretrained(full_model, init_path)

    full_model.crop_size = 224

    return full_model


@register_model
def Swin_ViCCT_large_22k(init_path=None, pretrained_cc=False, **kwargs):
    """ Swin-L @ 224x224, trained ImageNet-22k """

    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=192, depths=(2, 2, 18), num_heads=(6, 12, 24),
        **_clean_kwargs(kwargs))

    base_model = _create_swin_transformer('swin_large_patch4_window7_224_in22k', pretrained=False, **model_kwargs)

    if init_path and not pretrained_cc:
        base_model = init_model_state(base_model, init_path)

    full_model = DistilledRegressionTransformer(base_model, **model_kwargs)

    if init_path and pretrained_cc:
        full_model = load_pretrained(full_model, init_path)

    full_model.crop_size = 224

    return full_model


# ============================================================================================ #
#                               UTILITY FUNCTIONS TO LOAD WEIGHTS                              #
# ============================================================================================ #

def init_model_state(model, init_path):
    """ Loads ImageNet-pretrained backbone weights into `model`, skipping anything that
    doesn't exist in the checkpoint (e.g. because we dropped the classifier head / last stage). """

    if init_path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(init_path, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(init_path, map_location='cpu')

    pretrained_state = checkpoint['model'] if 'model' in checkpoint else checkpoint

    missing, unexpected = model.load_state_dict(pretrained_state, strict=False)

    print(f"Loaded backbone weights from '{init_path}'")
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    return model


def load_pretrained(model, init_path):
    """ Loads a fully pretrained ViCCT crowd-counting checkpoint (backbone + regression head). """

    resume_state = torch.load(init_path, map_location=torch.device('cpu'))
    state_dict = resume_state['state_dict'] if 'state_dict' in resume_state else resume_state
    model.load_state_dict(state_dict)

    return model
