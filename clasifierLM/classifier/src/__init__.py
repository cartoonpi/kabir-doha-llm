# Makes `src` a package so scripts run as modules: `python -m src.gen_requests`.
# Running as a module (instead of `python src/gen_requests.py`) lets files import
# each other as `from src.llm_client import complete`, from the repo root.
