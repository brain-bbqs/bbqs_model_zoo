"""Commandline interface for the bbqs_model_zoo package.

Examples:
    bbqs-zoo --help

"""
import click

from commands import fit, init, ls, predict, preprocess


@click.group("bbqs-zoo-cli")
def cli() -> None:
    """A collection of neuroimaging deep learning models."""
    return


cli.add_command(init)
cli.add_command(ls)
cli.add_command(fit)
cli.add_command(predict)
cli.add_command(preprocess)

def main():
   cli(prog_name="bbqs-zoo-cli")

if __name__ == "__main__":
    main()
