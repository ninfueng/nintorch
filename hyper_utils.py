import logging
import signal
import subprocess
import sys
from typing import List, Optional, Sequence, Union

import optuna
from nincore import gprint, rprint
from optuna.trial import Trial

logger = logging.getLogger(__name__)
trial_counter = 0


def run_cmd(cmd: str, getline: int = -2) -> Optional[float]:
    """Run command and receive a stdout `getline` line.

    Example:
    >>> run_cmd("python main.py")
    """
    assert isinstance(cmd, str)
    assert isinstance(getline, int)

    if cmd.find("python3") > -1:
        PYTHON = sys.executable
        cmd = cmd.replace("python3", PYTHON)

    elif cmd.find("python") > -1:
        PYTHON = sys.executable
        cmd = cmd.replace("python", PYTHON)

    cmd = cmd.split(" ")
    stdout = subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode()

    try:
        result = float(stdout.split("\n")[getline])
    except (IndexError, ValueError):
        # Catche the error when the float converting is not possible or
        # stdout information have lower lines than `getline`.
        result = None

    return result


class Suggestion:
    """A wrapper of `optuna.Trial` to provide suggestion to a trial.

    Created to converting outputs from `suggest_*` to cmd to parse with `argparse`.
    """

    def __init__(
        self,
        name: str,
        typename: str,
        amin: Optional[Union[float, int]] = None,
        amax: Optional[Union[float, int]] = None,
        choices: Optional[Sequence[Union[None, bool, int, float, str]]] = None,
    ) -> None:
        assert typename in ("float", "categorical", "int", "uniform", "log_uniform")
        self.name = name
        self.typename = typename
        self.amin = amin
        self.amax = amax
        self.choices = choices

    def suggest(self, trial: optuna.Trial) -> float:
        if self.typename == "float":
            suggest = trial.suggest_float(self.name, self.amin, self.amax)
        elif self.typename == "categorical":
            suggest = trial.suggest_categorical(self.name, self.choices)
        elif self.typename == "int":
            suggest = trial.suggest_int(self.name, self.amin, self.amax)
        elif self.typename == "uniform":
            suggest = trial.suggest_uniform(self.name, self.amin, self.amax)
        elif self.typename == "log_uniform":
            suggest = trial.suggest_loguniform(self.name, self.amin, self.amax)
        else:
            raise NotImplementedError(
                'Expect `typename` in ("float", "categorical", "int", "uniform", "log_uniform").'
                f"Your `typename`: {self.typename}."
            )
        return suggest

    def gen_cmd(self, suggest: Union[int, float]) -> str:
        args = f" --{self.name} {suggest}"
        return args


def run_script(trial: Trial, cmd: str, suggestions: List[Suggestion], timeout: Optional[int] = None) -> Optional[float]:
    """
    Args:
        trial (Trial): Optuna Trial objects to give suggested hyper-parameters
        script (str): name of script to run.
        timeout (int): seconds to interrupt a given trial.

    Example:
    >>> wrapped_run_script = lambda trial: run_script(trial, "python main.py")
    >>> study = optuna.create_study(study_name="study", direction="maximize")
    >>> study.optimize(wrapped_run_script, n_trials=n_trials)
    """
    assert cmd.find("python") > -1, "Cannot find `python` in `cmd`."
    global trial_counter

    for suggestion in suggestions:
        suggested = suggestion.suggest(trial)
        args = suggestion.gen_cmd(suggested)
        cmd += args
    cmd = cmd.replace("_", "-")
    gprint(f"Running command: `{cmd}`.")

    score = None
    if timeout is not None:
        signal.alarm(timeout)
        try:
            score = run_cmd(cmd)
        except TimeoutError:
            rprint(f"Timeout interrupt! using more than {timeout}, skipping {cmd}.")
        signal.alarm(0)

    else:
        score = run_cmd(cmd)

    gprint(f"Done trial#{trial_counter} with score: {score}")
    trial_counter += 1
    return score
