from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger

from experiments.tau_voice.exp.performance_analysis import extract_params_from_results
from tau2.data_model.simulation import Results
from tau2.utils.utils import DATA_DIR


def load_simulation_results(
    data_dir: Path, filter_domains: Optional[List[str]] = None
) -> List[Tuple[dict, Results]]:
    """
    Load all tau_voice simulation results from the given directory.

    Recursively searches for results.json files and extracts metadata from them.

    Args:
        data_dir: Directory containing simulation folders (searched recursively)
        filter_domains: Optional list of domains to include

    Returns:
        List of tuples (params_dict, Results)
    """
    results = []

    if not data_dir.exists():
        raise ValueError(f"Data directory {data_dir} does not exist.")

    # Find all results.json files recursively
    results_files = list(data_dir.rglob("results.json"))

    if not results_files:
        logger.warning(f"No results.json files found in {data_dir}")
        return results

    logger.info(f"Found {len(results_files)} results.json file(s)")

    for results_file in results_files:
        try:
            sim_results = Results.load(results_file)
            folder_path = results_file.parent

            # Extract params from the results object
            params = extract_params_from_results(sim_results, folder_path)

            # Filter by domain if specified
            if filter_domains and params["domain"] not in filter_domains:
                logger.debug(
                    f"Skipping {folder_path.name}: domain {params['domain']} not in filter"
                )
                continue

            results.append((params, sim_results))
            logger.info(
                f"Loaded: {params['domain']} / {params['speech_complexity']} / {params['llm']}"
            )
        except Exception as e:
            logger.error(f"Failed to load {results_file}: {e}")

    return results


path = DATA_DIR / "exp" / "pilot_exp_20tasks_openai_gemini"
results = load_simulation_results(path)
print(len(results))
