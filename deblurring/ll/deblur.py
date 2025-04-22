import numpy as np
import torch

from deblurring import architecture as arch


class deblur1:

    def __init__(self, model_path, device='cpu', upscale=4):
        self.device = torch.device(device)
        model = arch.MPRNet(in_c=3, out_c=3, n_feat=96, scale_unetfeats=48, scale_orsnetfeats=32, num_cab=8, kernel_size=3,
                 reduction=4, bias=False)
        model.load_state_dict(torch.load(model_path), strict=False)
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        self.model = model.to(device)

    def upscale(self, input):
        # read image and convert to 0-1 scale
        from torchvision.transforms import ToTensor
        totensor = ToTensor()
        img_tensor = totensor(input)
        img_LR = torch.unsqueeze(img_tensor , dim = 0 )
        if img_LR.shape[1] == 4 :
            img_LR =img_LR[:,:3,:,:]
        # img = cv2.imread(input, cv2.IMREAD_COLOR)
        # img = img * 1.0 / 255
        # img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        # img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(self.device)
        print(img_LR.shape)

        # neural network magic
        with torch.no_grad():
            result = self.model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
            result = np.transpose(result,(1,2,0)) *255
            # cv2.imwrite(output, result)
        return result