from timm.models.swin_transformer import _create_swin_transformer
import re
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
        # The pretrained ViCCT checkpoints were trained against an older timm where the Swin
        # backbone had a separate `pos_drop` module between patch_embed and the stages (so
        # `layers` sat at index 2: base_model.2.*). Modern timm folds pos_drop away entirely,
        # which would silently shift every stage's state_dict key by one and break loading.
        # We reinsert a no-op placeholder at index 1 purely to keep that indexing/key layout
        # intact, so pretrained checkpoints still load with the expected key names.
        self.base_model = nn.Sequential(base_model.patch_embed, nn.Identity(), base_model.layers)
        self.regression_head = ViCCTRegressionHead(224, kwargs['embed_dim'] * 4)

    def forward(self, x):
        x = self.base_model(x)
        den = self.regression_head(x)
        return den


# ============================================================================================ #
#                                           THE MODELS                                         #
# ============================================================================================ #

_TIMM_INTERNAL_KWARGS = {
    'pretrained', 'pretrained_cfg', 'pretrained_cfg_overlay',
    'cache_dir', 'scriptable', 'exportable', 'no_jit',
}


def _clean_kwargs(kwargs):
    """ timm's `create_model` (a) auto-injects control kwargs like pretrained=False,
    pretrained_cfg=None, cache_dir=None into every call, and (b) the notebook itself passes
    drop_rate/drop_path_rate/drop_block_rate=None. Modern timm's Swin implementation chokes
    on None floats, and forwarding timm's own control kwargs back into
    `_create_swin_transformer` collides with the `pretrained=False` we set explicitly below.
    So: drop timm's internal control kwargs entirely, and drop anything left set to None. """
    return {
        k: v for k, v in kwargs.items()
        if k not in _TIMM_INTERNAL_KWARGS and v is not None
    }


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


_DOWNSAMPLE_KEY_RE = re.compile(r'^(base_model\.2)\.(\d+)\.downsample\.(.*)$')


def _remap_downsample_keys(state_dict):
    """ Older timm placed each stage's PatchMerging *after* that stage's blocks, so
    layers[i].downsample described the transition INTO stage i+1 (e.g. layers[1].downsample
    took you from stage 1's dim to stage 2's dim). Modern timm places that same transition
    at the *start* of the following stage instead: layers[i+1].downsample. The transformer
    blocks themselves keep identical per-stage indexing either way (their input dim doesn't
    change), so we only need to shift the downsample keys by one stage index. """
    remapped = {}
    for k, v in state_dict.items():
        m = _DOWNSAMPLE_KEY_RE.match(k)
        if m:
            prefix, stage_idx, rest = m.groups()
            k = f'{prefix}.{int(stage_idx) + 1}.downsample.{rest}'
        remapped[k] = v
    return remapped


def load_pretrained(model, init_path):
    """ Loads a fully pretrained ViCCT crowd-counting checkpoint (backbone + regression head). """

    resume_state = torch.load(init_path, map_location=torch.device('cpu'))
    state_dict = resume_state['state_dict'] if 'state_dict' in resume_state else resume_state
    state_dict = _remap_downsample_keys(state_dict)

    # strict=False because older timm checkpoints also stored `attn_mask` /
    # `relative_position_index` buffers, which modern timm computes on the fly instead of
    # persisting. Those aren't learned weights, so skipping them is safe. Anything else
    # unexpectedly missing/mismatched gets printed so it's not a silent failure.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    real_missing = [k for k in missing]
    real_unexpected = [k for k in unexpected if not (k.endswith('attn_mask') or k.endswith('relative_position_index'))]

    if real_missing:
        print(f"WARNING: {len(real_missing)} expected weight(s) were not found in the checkpoint:")
        for k in real_missing[:10]:
            print("   ", k)
    if real_unexpected:
        print(f"WARNING: {len(real_unexpected)} unexpected weight(s) in the checkpoint were ignored:")
        for k in real_unexpected[:10]:
            print("   ", k)
    if not real_missing and not real_unexpected:
        print("Checkpoint loaded cleanly (only non-persistent geometry buffers were skipped, as expected).")

    return model
