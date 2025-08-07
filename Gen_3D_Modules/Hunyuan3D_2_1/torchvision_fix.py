# Torchvision compatibility fix for functional_tensor module
# This file helps resolve compatibility issues between different torchvision versions

import sys
import torchvision

def fix_torchvision_functional_tensor():
    """
    Fix torchvision.transforms.functional_tensor import issue
    """
    try:
        # Check if the module exists in the expected location
        import torchvision.transforms.functional_tensor
        print("torchvision.transforms.functional_tensor is available")
        return True
    except ImportError:
        print("torchvision.transforms.functional_tensor not found, applying compatibility fix...")
        
        try:
            # Create a mock functional_tensor module with the required functions
            import torch
            import torchvision.transforms.functional as F
            
            class FunctionalTensorMock:
                """Mock module to replace functional_tensor"""
                
                @staticmethod
                def rgb_to_grayscale(img, num_output_channels=1):
                    """Convert RGB image to grayscale"""
                    if hasattr(F, 'rgb_to_grayscale'):
                        return F.rgb_to_grayscale(img, num_output_channels)
                    else:
                        # Fallback implementation
                        if len(img.shape) == 4:  # Batch of images
                            weights = torch.tensor([0.299, 0.587, 0.114], 
                                                   device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
                        else:  # Single image
                            weights = torch.tensor([0.299, 0.587, 0.114], 
                                                   device=img.device, dtype=img.dtype).view(3, 1, 1)
                        
                        grayscale = torch.sum(img * weights, dim=-3, keepdim=True)
                        
                        if num_output_channels == 3:
                            if len(img.shape) == 4:
                                grayscale = grayscale.repeat(1, 3, 1, 1) 
                            else:
                                grayscale = grayscale.repeat(3, 1, 1)
                        
                        return grayscale
                
                @staticmethod
                def resize(img, size, interpolation=2, antialias=None):
                    """Resize function wrapper"""
                    try:
                        from torchvision.transforms.v2.functional import resize as v2_resize
                        return v2_resize(img, size, 
                                         interpolation=interpolation, 
                                         antialias=antialias)
                    except ImportError:
                        if hasattr(F, 'resize'):
                            return F.resize(img, size, interpolation=interpolation)
                        else:
                            import torch.nn.functional as torch_F
                            if isinstance(size, int):
                                size = (size, size)
                            return torch_F.interpolate(
                                img.unsqueeze(0) if len(img.shape) == 3 else img,
                                size=size, mode='bilinear', align_corners=False)
                
                def __getattr__(self, name):
                    """Fallback to regular functional module"""
                    if hasattr(F, name):
                        return getattr(F, name)
                    else:
                        # Try v2.functional
                        try:
                            import torchvision.transforms.v2.functional as v2_F
                            if hasattr(v2_F, name):
                                return getattr(v2_F, name)
                        except ImportError:
                            pass
                        
                        raise AttributeError(f"'{name}' not found in functional_tensor mock")
            
            # Create the mock module instance
            functional_tensor_mock = FunctionalTensorMock()
            
            # Monkey patch the old location
            sys.modules['torchvision.transforms.functional_tensor'] = functional_tensor_mock
            print("Applied compatibility fix: created functional_tensor mock module")
            return True
            
        except Exception as e:
            print(f"Failed to create functional_tensor mock: {e}")
            return False

def apply_fix():
    """Apply the torchvision compatibility fix"""
    print(f"Torchvision version: {torchvision.__version__}")
    return fix_torchvision_functional_tensor()

if __name__ == "__main__":
    apply_fix() 
    