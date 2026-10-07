"""build_model(cfg) instantiates any network from a config dictionary."""


def build_model(cfg):
    name = cfg["model"]
    if name == "ssm_fpp":
        from .ssm_fpp import SSMFPP
        return SSMFPP(in_channels=cfg["in_channels"], out_channels=cfg["out_channels"],
                      dim=cfg["dim"], num_blocks=tuple(cfg["num_blocks"]),
                      d_state=cfg["d_state"], d_conv=cfg["d_conv"], expand=cfg["expand"],
                      mixer=cfg.get("mixer", "ssm"))
    if name == "hidnet":
        from .hidnet import HiDNet
        return HiDNet(in_channels=cfg["in_channels"], out_channels=cfg["out_channels"],
                      base_filters=cfg["base_filters"], dropout=cfg["dropout"],
                      alpha=cfg["alpha"])
    if name == "nafnet":
        from .nafnet import NAFNet
        return NAFNet(in_channels=cfg["in_channels"], out_channels=cfg["out_channels"],
                      width=cfg["width"], enc_blocks=tuple(cfg["enc_blocks"]),
                      middle_blocks=cfg["middle_blocks"], dec_blocks=tuple(cfg["dec_blocks"]),
                      drop_prob=cfg["drop_prob"])
    if name == "restormer":
        from .restormer import Restormer
        return Restormer(in_channels=cfg["in_channels"], out_channels=cfg["out_channels"],
                         dim=cfg["dim"], num_blocks=tuple(cfg["num_blocks"]),
                         num_heads=tuple(cfg["num_heads"]),
                         ffn_expansion=cfg["ffn_expansion"], bias=cfg["bias"])
    raise ValueError(f"unknown model {name!r}")
