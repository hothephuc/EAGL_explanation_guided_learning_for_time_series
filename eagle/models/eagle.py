import torch
from torch import nn
import torch.nn.functional as F


class Eagle(nn.Module):
    """Explanation-guided wrapper around a CNN classifier.

    The CAM of the predicted class is computed from the activations of `target_layer` and the
    weights of the classifier's `fc` layer, turned into a soft mask
    `sigmoid((CAM - threshold) * omega)` and aligned with the expert mask through an MSE loss:

        L_total = L_cls + gamma * L_exp
    """

    def __init__(self, model, num_classes=9, target_layer=None):
        super(Eagle, self).__init__()
        self.bmodel = model
        self.num_classes = num_classes
        self.target_layer = target_layer
        self.activations = None
        self._hook_handle = None
        if self.target_layer is not None:
            self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
        if self._hook_handle is not None:
            self._hook_handle.remove()
        self._hook_handle = self.target_layer.register_forward_hook(forward_hook)

    def _classifier_weight(self):
        base_model = self.bmodel.module if isinstance(self.bmodel, nn.DataParallel) else self.bmodel
        return base_model.fc.weight  # shape: num_classes x C

    def forward(self, x, target=None, mask_x=None, use_gain=False, class_weights=None, threshold=0.5, omega=10, gamma=0.1):
        if target is None:
            target = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        logits = self.bmodel(x)
        cl_loss = F.cross_entropy(logits, target, weight=class_weights)
        if not use_gain:
            return cl_loss, logits, None, None, None, 0, 0

        B = x.size(0)
        pred_c = torch.argmax(logits, dim=1)
        A = self.activations  # feature maps: B x C x L'
        weights = self._classifier_weight()[pred_c]  # shape: B x C

        # compute CAM: weighted sum of activations
        cam = torch.sum(weights.unsqueeze(2) * A, dim=1)  # shape: B x L'
        cam = F.relu(cam)

        cam = F.interpolate(cam.unsqueeze(1), size=x.size(-1), mode='linear', align_corners=False)
        cam = cam.squeeze(1)  # shape: B x L

        # Normalize the CAM to [0, 1]
        flat_cam = cam.view(B, -1)
        cam_min = flat_cam.min(dim=1, keepdim=True)[0]
        cam_max = flat_cam.max(dim=1, keepdim=True)[0]
        cam = (cam - cam_min) / ((cam_max - cam_min) + 1e-7)

        mask = torch.sigmoid((cam - threshold) * omega)

        # Only the regions marked by the expert (mask == 1) are supervised: everywhere else the
        # target is replaced by the model's own mask, so samples without an expert mask add no loss.
        mask_x = mask.detach().clone() if mask_x is None else mask_x.clone()
        indices = mask_x == 0
        mask_x[indices] = mask[indices]
        att_loss = F.mse_loss(mask, mask_x)
        loss = cl_loss + gamma * att_loss
        return loss, logits, None, cam, mask_x, cl_loss.item(), att_loss.item()
