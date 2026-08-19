from cmr.framework import MethodCapabilities, MethodSpec, register_method

from .method import BCGTSHMethod, BCGTSHMethodConfig, load_bc_gtsh_method_config

BC_GTSH_SPEC = MethodSpec(
    method_id="bc-gtsh",
    display_name="BC-GTSH",
    config_loader=load_bc_gtsh_method_config,
    factory=BCGTSHMethod,
    capabilities=MethodCapabilities(
        continuous_embeddings=True,
        checkpoint=True,
        train_log=True,
        required_modalities=("image", "text"),
        supports_independent_modalities=True,
        supported_supervision=("full",),
    ),
)

register_method(BC_GTSH_SPEC)

__all__ = ["BCGTSHMethod", "BCGTSHMethodConfig", "BC_GTSH_SPEC"]

