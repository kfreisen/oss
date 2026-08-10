import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import washbook

    return washbook, mo


@app.cell
def _(mo):
    mo.md(
        """
        # washbook — quickstart

        This package is at `0.0.1.dev0`: the name is reserved and the release
        pipeline works, but the implementation has not landed yet. This notebook
        exists so the docs build, the export step, and the CI check that executes
        every example are all exercised from day one rather than bolted on later.

        It is replaced with a real walkthrough in `0.1.0`.
        """
    )
    return


@app.cell
def _(washbook, mo):
    mo.md(f"Installed version: `{washbook.__version__}`")
    return


if __name__ == "__main__":
    app.run()
