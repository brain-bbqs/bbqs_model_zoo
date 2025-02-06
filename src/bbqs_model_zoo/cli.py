import click

from commands import fit, init, ls, predict, preprocess


@click.group()
def main() -> None:
    """A collection of neuroimaging deep learning models."""
    return


main.add_command(init)
main.add_command(ls)
main.add_command(fit)
main.add_command(predict)
main.add_command(preprocess)

# for debugging purposes
if __name__ == "__main__":
    main()
