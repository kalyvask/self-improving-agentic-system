from wdp.benchmarks.base import Benchmark
from wdp.benchmarks.arithmetic import (
    ArithmeticBenchmark,
    ArithmeticVerifier,
    safe_eval,
    split,
)

__all__ = [
    "Benchmark",
    "ArithmeticBenchmark",
    "ArithmeticVerifier",
    "safe_eval",
    "split",
]

# tau-bench is an optional heavy dependency. Only export the adapter if it is
# installed, so importing this package stays cheap and works offline.
try:
    from wdp.benchmarks.taubench import (
        TauBenchBenchmark,
        TauReActExecutor,
        TauTerminalVerifier,
    )
except ImportError:
    pass
else:
    __all__ += ["TauBenchBenchmark", "TauReActExecutor", "TauTerminalVerifier"]
