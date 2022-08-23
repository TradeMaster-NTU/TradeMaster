#!/usr/bin/env python3
# coding=gbk
import json
from typing import List
import math
import argparse
from levels import InnerLevel, OuterLevel, CompassEntry

# Constants
D = 6  # Number of protocol dimensions
EV = 16  # Number of evaluation measures
A = 360 / D  # Angle between method axes
B = 360 / EV  # Angle between evaluation measure axes


def parse_arguments():
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser(
        description="CLEVA-Compass Generator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--template",
        default="Compass/generate/compass/blank.tex",
        help="Tikz template file.",
    )
    parser.add_argument(
        "--output",
        default="Compass/generate/compass/filled.tex",
        help="Tikz filled output file.",
    )
    parser.add_argument(
        "--data",
        default="Compass/generate/compass/data.json",
        help="Entries as JSON file.",
    )

    return parser.parse_args()


def mapcolor(color):
    """Maps the given simple colors string to a specific color for latex."""
    return {
        "magenta": "magenta",
        "green": "green!50!black",
        "blue": "blue!70!black",
        "orange": "orange!90!black",
        "cyan": "cyan!90!black",
        "brown": "brown!90!black",
    }[color]


def insert_legend(template, entries):
    """Insert the CLEVA-Compass legend below the compass."""

    # Skip if no entries are given (else the empty tabular will produce compile errors)
    if len(entries) == 0:
        return template

    # Compute number of rows/columns with max. three elements per row
    n_rows = math.ceil(len(entries) / 6)
    n_cols = 6 if len(entries) >= 6 else len(entries)

    # Begin legend tabular
    legend_str = ""
    legend_str += r"\begin{tabular}{" + " ".join(["l"] * n_cols) + "} \n"

    for i, e in enumerate(entries):
        # x/y coordinates of the entry
        x = i % 6
        y = round(i // 6)

        # Actual entry which uses \lentry defined in the tikz template
        legend_str += r"\lentry{" + mapcolor(e.color) + "}{" + e.label + "}"

        # Depending on last column/row
        is_last_column = x == n_cols - 1
        is_last_row = y == n_rows - 1
        if not is_last_column:
            # Add & for next element in row
            legend_str += r" & "
        else:
            if not is_last_row:
                # Add horizontal space if there is another row
                legend_str += " \\\\[0.15cm] \n"
            else:
                # Add no horizontal space if this is the last row
                legend_str += " \\\\ \n"

    # End legend tabular
    legend_str += "\end{tabular} \n"

    # Replace the generated string in template
    template = template.replace("%-$LEGEND$", legend_str)
    return template


def insert_outer_level(template, entries: List[CompassEntry]):
    """Insert outer level attributes."""
    oc_str = ""
    M = len(entries)
    for e_idx, e in enumerate(entries):
        # Add comment for readability
        s = "% Entry for: " + e.label + "\n"

        # For each outer level attribute
        for ol_idx, has_attribute in enumerate(e.outer_level):
            # If attribute is not present, skip and leave white
            if not has_attribute:
                continue
            angle_start = str(ol_idx * B + e_idx * B / M)
            angle_end = str(ol_idx * B + (e_idx + 1) * B / M)

            # Invert stripe direction when in the lower half (index larger than 7)
            if ol_idx > 7:
                angle_start, angle_end = angle_end, angle_start

            shell = e.color + "shell"
            s += (
                "\pic at (0,0){strip={\Instrip,"
                + angle_start
                + ","
                + angle_end
                + ","
                + shell
                + ", black, {}}};\n"
            )
        oc_str += s + "\n"

    template = template.replace("%-$OUTER-CIRCLE$", oc_str)
    return template


def insert_inner_level(template, entries: List[CompassEntry]):
    """Insert inner level path connections."""
    ir_str = ""
    for e in entries:
        path = " -- ".join(f"(D{i+1}-{irv})" for i, irv in enumerate(e.inner_level))
        ir_str += f"\draw [color={mapcolor(e.color)},line width=1.5pt,opacity=0.6, fill={mapcolor(e.color)}!10, fill opacity=0.4] {path} -- cycle;\n"

    template = template.replace("%-$INNER-CIRCLE$", ir_str)
    return template


def insert_number_of_methods(template, entries: List[CompassEntry]):
    """Insert number of methods as newcommand \M."""
    n_methods_str = r"\newcommand{\M}{" + str(len(entries)) + "}"
    template = template.replace("%-$NUMBER-OF-METHODS$", n_methods_str)
    return template


def read_json_entries(entries_json):
    """Read the compass entries from a json file."""
    entries = []
    for d in entries_json:
        dil = d["inner_level"]
        dol = d["outer_level"]
        entry = CompassEntry(
            color=d["color"],
            label=d["label"],
            inner_level=InnerLevel(
                Proftability=dil["Proftability"],
                Risk_Control=dil["Risk_Control"],
                University=dil["University"],
                Diversity=dil["Diversity"],
                Reliability=dil["Reliability"],
                Explainability=dil["Explainability"],
            ),
            outer_level=OuterLevel(
                alpha_decay=dol["alpha_decay"],
                profit=dol["profit"],
                extreme_market=dol["extreme_market"],
                risk_adjusted=dol["risk_adjusted"],
                risk=dol["risk"],
                time_scale=dol["time_scale"],
                assert_type=dol["assert_type"],
                country=dol["country"],
                rolling_window=dol["rolling_window"],
                correlation=dol["correlation"],
                entropy=dol["entropy"],
                t_SNE=dol["t_SNE"],
                rank_order=dol["rank_order"],
                variability=dol["variability"],
                profile=dol["profile"],
                equity_curve=dol["equity_curve"],
            ),
        )
        entries.append(entry)
    return entries


def generate_random_entries():
    import numpy as np

    np.random.seed(0)

    entries = []
    for i in range(6):
        entries.append(
            CompassEntry(
                color=np.random.choice(["magenta", "cyan", "green", "orange", "brown", "blue"]),
                label="Method " + str(i),
                inner_level=InnerLevel(
                    Proftability=np.random.randint(100),
    Risk_Control=np.random.randint(100),
    University=np.random.randint(100),
    Diversity=np.random.randint(100),
    Reliability=np.random.randint(100),
    Explainability=np.random.randint(100),
                ),
                outer_level=OuterLevel(
                    alpha_decay=bool(np.random.randint(2)),
                    profit=bool(np.random.randint(2)),
                    extreme_market=bool(np.random.randint(2)),
                    risk_adjusted=bool(np.random.randint(2)),
                    risk=bool(np.random.randint(2)),
                    time_scale=bool(np.random.randint(2)),
                    assert_type=bool(np.random.randint(2)),
                    country=bool(np.random.randint(2)),
                    rolling_window=bool(np.random.randint(2)),
                    correlation=bool(np.random.randint(2)),
                    entropy=bool(np.random.randint(2)),
                    t_SNE=bool(np.random.randint(2)),
                    rank_order=bool(np.random.randint(2)),
                    variability=bool(np.random.randint(2)),
                    profile=bool(np.random.randint(2)),
                    equity_curve=bool(np.random.randint(2)),
                ),
            )
        )

    return entries


def fill_template(template_path, entries):
    template_path = template_path
    with open(template_path) as f:
        template = "".join(f.readlines())

    # Replace respective parts in template
    output = template
    output = insert_legend(output, entries)
    output = insert_outer_level(output, entries)
    output = insert_inner_level(output, entries)
    output = insert_number_of_methods(output, entries)

    return output


if __name__ == "__main__":
    args = parse_arguments()

    # Read the compass entry from the given json data file
    entries_json = json.load(open(args.data))["entries"]
    entries = read_json_entries(entries_json)
    # entries = generate_random_entries()

    # Read template content
    output = fill_template(args.template, entries)

    # Write output to the desired destination
    with open(args.output, "w") as f:
        f.write(output)