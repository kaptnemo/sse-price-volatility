"""
Shared matplotlib configuration for Chinese font support.

Import this module in any file that produces matplotlib figures to ensure
Chinese characters render correctly instead of showing as boxes.

Usage
-----
    import src.plot_config  # noqa: F401  (side-effect import)
"""

from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# Bundled font shipped with this project (assets/fonts/)
_BUNDLED_FONT = Path(__file__).resolve().parent.parent / "assets" / "fonts" / "NotoSansCJK-Regular.ttc"


def _setup_chinese_font() -> None:
    """Configure matplotlib to use a CJK font.

    Priority:
    1. Bundled NotoSansSC font in ``assets/fonts/``.
    2. Any CJK font already installed on the system.
    """
    # --- 1. Try bundled font first ---
    if _BUNDLED_FONT.exists():
        fm.fontManager.addfont(str(_BUNDLED_FONT))
        prop = fm.FontProperties(fname=str(_BUNDLED_FONT))
        font_name = prop.get_name()
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [font_name] + plt.rcParams["font.sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False
        return

    # --- 2. Fall back to system CJK fonts ---
    candidates = [
        "Noto Sans CJK SC",
        "Noto Sans CJK TC",
        "Noto Sans CJK JP",
        "AR PL UMing CN",
        "AR PL UKai CN",
        "WenQuanYi Micro Hei",
        "SimHei",
        "Microsoft YaHei",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((name for name in candidates if name in available), None)

    if chosen:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [chosen] + plt.rcParams["font.sans-serif"]
    else:
        import warnings
        warnings.warn(
            "No CJK font found; Chinese characters may not display correctly. "
            f"Bundled font not found at {_BUNDLED_FONT}. Tried system fonts: {candidates}",
            UserWarning,
            stacklevel=2,
        )

    plt.rcParams["axes.unicode_minus"] = False


_setup_chinese_font()
