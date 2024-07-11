from ..utils.ops import scale_tensor
from ..utils.misc import get_device


class LPIPS:
    def __init__(self):
        import lpips
        self.model = lpips.LPIPS(net="vgg").to(get_device())
        self.model.eval()
        for params in self.model.parameters():
            params.requires_grad = False
        self.model_input_range = (-1, 1)

    def __call__(self, x1, x2, return_layers=False, input_range=(0, 1)):
        x1 = scale_tensor(x1, input_range, self.model_input_range)
        x2 = scale_tensor(x2, input_range, self.model_input_range)
        return self.model(x1, x2, retPerLayer=return_layers, normalize=False)
