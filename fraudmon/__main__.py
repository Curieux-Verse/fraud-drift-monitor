"""Command line entry point: python -m fraudmon <step>."""

import argparse

from . import pipeline

STEPS = {
    "data": pipeline.step_data,
    "train": pipeline.step_train,
    "compare": pipeline.step_compare,
    "threshold": pipeline.step_threshold,
    "drift": pipeline.step_drift,
    "simulate": pipeline.step_simulate,
    "plots": pipeline.step_plots,
    "summary": pipeline.step_summary,
    "evaluate": pipeline.step_evaluate,
    "all": pipeline.step_all,
}


def main():
    parser = argparse.ArgumentParser(
        prog="python -m fraudmon",
        description="Fraud detection with drift monitoring. `all` runs data, train and evaluate.",
    )
    parser.add_argument("step", choices=STEPS)
    STEPS[parser.parse_args().step]()


if __name__ == "__main__":
    main()
