"""Run many Sim configs in worker processes; the parent logs to MLflow."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from concurrent.futures import ProcessPoolExecutor


def _work(args):
    import mlflow_experiments as X
    from f1_weekend_model import ModelParams
    pdict, split, n_sims = args
    fc, pw, secs = X.evaluate(X.Sim(ModelParams(**pdict), n_sims=n_sims), split)
    return fc, pw, secs


def eval_many(pdicts, split, n_sims=3000, workers=8):
    jobs = [(p, split, n_sims) for p in pdicts]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_work, jobs))
