def to_latex_row(values, stds=None):
    text = [f"${v:.4f}$" for v in values]
    if stds is not None and not any([x is None for x in stds]):
        text = [f"{v[:-1]} \\pm {std:.4f}$" for (v, std) in zip(text, stds)]
    print(' & '.join(text))


def to_latex_table(values_all, stds_all=None):
    if stds_all is None:
        stds_all = [None] * len(values_all)

    for values, stds, in zip(values_all, stds_all):
        to_latex_row(values, stds)
