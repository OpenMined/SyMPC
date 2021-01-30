# stdlib
import glob
from pathlib import Path

# third party
import nbformat
import papermill as pm
import pytest

# lets start by finding all notebooks currently available in examples and subfolders
all_notebooks = (n for n in glob.glob("examples/**/*.ipynb", recursive=True))
basic_notebooks = (n for n in glob.glob("examples/*.ipynb"))
advanced_notebooks = (
    n for n in glob.glob("examples/advanced/**/*.ipynb", recursive=True)
)


# buggy notebooks with explanation what does not work
exclusion_list_notebooks = []

exclusion_list_folders = []


excluded_notebooks = []
for nb in all_notebooks:
    if Path(nb).name in exclusion_list_notebooks:
        excluded_notebooks += [nb]
for folder in exclusion_list_folders:
    excluded_notebooks += glob.glob(f"{folder}/**/*.ipynb", recursive=True)


@pytest.mark.parametrize(
    "notebook", sorted(set(basic_notebooks) - set(excluded_notebooks))
)
def test_notebooks_basic(notebook):
    """Test Notebooks in the tutorial root folder."""
    res = pm.execute_notebook(
        notebook,
        "/dev/null",
        parameters={
            "epochs": 1,
            "n_test_batches": 5,
            "n_train_items": 64,
            "n_test_items": 64,
        },
        kernel_name="python3",
        timeout=600,
    )
    assert isinstance(res, nbformat.notebooknode.NotebookNode)


@pytest.mark.parametrize(
    "notebook", sorted(set(advanced_notebooks) - set(excluded_notebooks))
)
def test_notebooks_advanced(notebook):
    res = pm.execute_notebook(
        notebook, "/dev/null", parameters={"epochs": 1}, timeout=600
    )
    assert isinstance(res, nbformat.notebooknode.NotebookNode)
