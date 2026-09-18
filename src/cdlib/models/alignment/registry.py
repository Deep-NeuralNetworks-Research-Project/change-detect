"""ALIGNMENT_REGISTRY — all alignment modules register here.

Register your alignment module with:
    @ALIGNMENT_REGISTRY.register("bounded")
    class BoundedAlignment(nn.Module):
        def forward(self, f1: Tensor, f2: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            '''Return (aligned_f1, aligned_f2, offset_map)'''
            ...
"""

from cdlib.utils.registry import Registry

ALIGNMENT_REGISTRY = Registry("ALIGNMENT")
