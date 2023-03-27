from omegaconf import DictConfig, ListConfig


def get_dotlist_params(cfg, cond=None):
    dotlist = {}

    def gather(_cfg):
        if isinstance(_cfg, ListConfig):
            raise ValueError("ListConfig not supported as first container.")
        for key in _cfg:
            dotlist_key = _cfg._get_full_key(key)  # noqa
            if isinstance(_cfg[key], DictConfig):
                gather(_cfg[key])
            elif cond is None or cond(dotlist_key, _cfg[key]):
                dotlist[dotlist_key] = _cfg[key]

    gather(cfg)
    return dotlist
