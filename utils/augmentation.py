import torch

class AugmentationGAN(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, model, noise_sigma):
        self.model = model
        self.noise_sigma = noise_sigma

    def __call__(self, input_data):
        image, class_label = input_data

        # Ecode the image
        encoding = model.encode(img)

        # Generate noise
        noise = torch.randn(encoding.shape) * noise_sigma

        # Decode image and generate augmented image
        out_image = model.decode(encoding + noise)

        return out_image, class_label