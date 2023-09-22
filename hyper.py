"""Hyper parameters tuning scripts."""
import argparse

import joblib
import optuna
from nincore import gprint

from hyper_utils import Suggestion, run_script

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Hyper searching script.')
    parser.add_argument('--cmd', type=str, default='python main.py')
    parser.add_argument('--n-trials', type=int, default=100)
    parser.add_argument('--study-name', type=str, default='hyper-cifar10')
    parser.add_argument('--optimize-direction', type=str, default='maximize')
    args = parser.parse_args()

    suggests = [
        Suggestion('lr', 'float', amin=1e-2, amax=1e-1),
        Suggestion('weight_decay', 'float', amin=1e-6, amax=1e-3),
    ]

    wrapped_run_script = lambda trial: run_script(trial, args.cmd, suggests)
    study = optuna.create_study(study_name=args.study_name, direction=args.optimize_direction)
    study.optimize(wrapped_run_script, n_trials=args.n_trials)

    trial = study.best_trial
    gprint(f'\nBest trial value: {trial.value}')
    for key, value in trial.params.items():
        gprint(f'{key}: {value}')

    joblib.dump(study, f'{args.study_name}.pkl')
    df = study.trials_dataframe()
    df.to_csv(f'{args.study_name}.csv', index=None)
