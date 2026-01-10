# analysis/logs.py
from analysis.log_parsing import extract_observational_lines


def finalize_training_log(logger, handler, log_stream):
    logger.removeHandler(handler)
    full_log = log_stream.getvalue()
    return extract_observational_lines(full_log)
