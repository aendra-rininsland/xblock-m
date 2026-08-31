"""
Single source of truth for how an image becomes a model input.

Both the serving worker and the pipeline harvester import from here so that they
cannot drift apart. These values must stay identical to the validation transform
in the training notebook (xblock-notebooks/xblock-m-timm.ipynb):

    T.Resize((224,224))              # aspect-ratio SQUASH, not shortest-side
    T.ToTensor()
    T.Normalize(mean=0.5, std=0.5)

Two traps worth knowing about before changing anything here:

1. The published checkpoint's config.json advertises ImageNet mean/std
   (0.485/0.456/0.406). That is inherited from swin_s3_base_224.ms_in1k and is
   NOT what the model was trained with. Anything that builds its transform from
   the hub config -- timm's resolve_data_config(), for instance -- will silently
   mis-normalise every image. Use this module instead.

2. Resize with a (h, w) tuple squashes the aspect ratio. It is not equivalent to
   T.Resize(224), which resizes the shortest side and preserves aspect. timm's
   create_transform() cannot express the squash, which is the other way this
   drifts.
"""
import torchvision.transforms as T

IMG_SIZE = (224, 224)
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)


def build_transform():
    """The validation/serving transform. Matches valid_tfms in the notebook."""
    return T.Compose([
        T.Resize(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])
