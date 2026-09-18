"""LOSS_REGISTRY — all loss functions register here.

Register your loss with:
    @LOSS_REGISTRY.register("bce_dice")
    class BCEDiceLoss(nn.Module):
        def compute(self, outputs: dict, batch: dict) -> dict:
            '''Return {"loss": Tensor[], "loss/bce": ..., "loss/dice": ..., ...}'''
            ...
"""

from cdlib.utils.registry import Registry

LOSS_REGISTRY = Registry("LOSS")
