# analysis/log_parsing.py

def extract_observational_lines(full_log: str) -> dict:
    lines = full_log.splitlines()
    parsed = {
        "token_filtering": [],
        "config": [],
        "progress": [],
        "convergence": [],
        "perplexity": [],
        "final": []
    }

    for line in lines:
        if (
            "built Dictionary<" in line
            or "discarding" in line
            or "keeping" in line
            or "resulting dictionary" in line
        ):
            parsed["token_filtering"].append(line)

        elif (
            "using symmetric" in line
            or "running online LDA training" in line
            or "training LDA model" in line
        ):
            parsed["config"].append(line)

        elif line.startswith("PROGRESS:"):
            parsed["progress"].append(line)

        elif "documents converged" in line:
            parsed["convergence"].append(line)

        elif "perplexity estimate" in line:
            parsed["perplexity"].append(line)

        elif "LdaMulticore lifecycle event" in line:
            parsed["final"].append(line)

    return parsed
