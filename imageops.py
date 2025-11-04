import os
import PIL

from typing import Union, Optional, Callable


# adapted from huggingface/diffusers Apache 2.0
def load_image(
    image: Union[str, PIL.Image.Image],
    convert_method: Optional[Callable[[PIL.Image.Image], PIL.Image.Image]] = None,
) -> PIL.Image.Image:
    """
    Loads `image` to a PIL Image.

    Args:
        image (`str` or `PIL.Image.Image`):
            The image to convert to the PIL Image format.
        convert_method (Callable[[PIL.Image.Image], PIL.Image.Image], *optional*):
            A conversion method to apply to the image after loading it. When set to `None` the image will be converted
            "RGB".

    Returns:
        `PIL.Image.Image`:
            A PIL Image.
    """
    if isinstance(image, str):
        if os.path.isfile(image):
            image = PIL.Image.open(image)
        else:
            raise ValueError(f"Incorrec path: {image} is not a valid path.")
    elif isinstance(image, PIL.Image.Image):
        image = image
    else:
        raise ValueError(
            "Incorrect format used for the image. Should be a local path, or a PIL image."
        )

    image = PIL.ImageOps.exif_transpose(image)

    if convert_method is not None:
        image = convert_method(image)
    else:
        image = image.convert("RGB")

    return image
