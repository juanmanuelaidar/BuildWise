from pathlib import Path

from app.modules.pricing.domain.exceptions import ForecastRuntimeError

CMDSTAN_VERSION = "cmdstan-2.38.0"


def importar_dependencias_forecast():
    try:
        import cmdstanpy
        import pandas as pd
        from prophet import Prophet
        from prophet.models import CmdStanPyBackend, IStanBackend
    except ImportError as exc:
        raise ForecastRuntimeError("Faltan dependencias para generar el forecast.") from exc
    return cmdstanpy, pd, Prophet, CmdStanPyBackend, IStanBackend


def configurar_cmdstan(cmdstanpy, CmdStanPyBackend, IStanBackend) -> None:
    cmdstan_global = Path.home() / ".cmdstan" / CMDSTAN_VERSION
    if not cmdstan_global.exists():
        # The pinned Prophet wheel includes its own precompiled model/runtime.
        # Use it without downloading or compiling an unrelated global version.
        import prophet

        bundled = Path(prophet.__file__).parent / "stan_model" / f"cmdstan-{CmdStanPyBackend.CMDSTAN_VERSION}"
        if not bundled.exists():
            raise ForecastRuntimeError("No se encontro CmdStan para correr Prophet.")
        cmdstan_global = bundled

    cmdstanpy.set_cmdstan_path(str(cmdstan_global))

    def fixed_init(self):
        cmdstanpy.set_cmdstan_path(str(cmdstan_global))
        IStanBackend.__init__(self)

    CmdStanPyBackend.__init__ = fixed_init
