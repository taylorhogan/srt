import os, sys, socket

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),  '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from configs.config_private import PrivateConfig
from configs.config_public import PublicConfig


# Keys in a cfg["machine"][hostname] entry that override a nested config value
# on the current host. Maps the machine-entry key -> (section, key) it replaces.
_MACHINE_OVERRIDES = {
    "image_dir": ("nina", "image_dir"),
    # Calibration lives at a C: path on the observatory PC and a mirrored Linux
    # path on the Spark. Without these three, calibration_paths_from_config
    # resolves to nothing off-Windows and every stack taken there is silently
    # uncalibrated — no error, just hot pixels and dark structure left in.
    "bias_dir": ("calibration", "bias_dir"),
    "dark_dir": ("calibration", "dark_dir"),
    "flat_root": ("calibration", "flat_root"),
}


def _apply_machine_overrides(cfg):
    """Overlay the current host's cfg["machine"][hostname] entry.

    Lets one config serve several machines with different directory layouts:
    the matching machine block's "image_dir" replaces cfg["nina"]["image_dir"]
    so every existing reader stays host-aware. Hosts with no entry keep the
    shared defaults.
    """
    machine = cfg.get("machine", {}).get(socket.gethostname())
    if not machine:
        return
    for src_key, (section, dst_key) in _MACHINE_OVERRIDES.items():
        if src_key in machine:
            cfg.setdefault(section, {})[dst_key] = machine[src_key]


def data():
     a = PublicConfig().data()
     b = PrivateConfig().data()
     a.update(b)
     _apply_machine_overrides(a)
     return a




if __name__ == "__main__":
    fc = data()
    print (fc)
