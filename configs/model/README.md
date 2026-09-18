# Model configs — DRAFT

`/configs/` is P2's territory per `CODEOWNERS` (Hydra config system,
root `configs/config.yaml` defaults list). That root doesn't exist yet,
so these files aren't Hydra-functional in isolation — they exist so the
Week 2 checklist item ("Register both in ENCODER_REGISTRY;
`configs/model/` entries") isn't left undone, and so P2 has an exact
shape to fold into the real hierarchy.

Each file uses Hydra's `_target_` instantiation convention
(`hydra.utils.instantiate(cfg)` calls the class with the remaining keys
as kwargs) — the kwargs match each registered class's `__init__`
exactly, so this is drop-in once a root config composes them in.
