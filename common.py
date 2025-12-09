from pathlib import Path


def create_results_folder():
    this_folder = Path(__file__).parent
    results_folder = this_folder / 'results'
    results_folder.mkdir(exist_ok=True)
    return results_folder
