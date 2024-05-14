def generate_latex_figure(fig_number, alaNum):
    # Define the figure number
    ala = ["Vulture","HighSpeed"]
    fig_label = f"{ala[alaNum-1][0]}{fig_number}Sl"
    # Generate the LaTeX code
    if fig_number <=2:
        degree = fig_number*10
    else: degree = fig_number
    latex_code = f"""
\\begin{{figure}}[h]
    \\centering
    \\begin{{subfigure}}{{0.49\\textwidth}}
        \\centering
        \\includegraphics[width=\\textwidth]{{Pics/{ala[alaNum-1]}{fig_number}SliceU.png}}
        \\caption{{Vista lateral, colors segons magnitud de velocitat.}}
        \\label{{fig:{fig_label}U}}
    \\end{{subfigure}}
    \\begin{{subfigure}}{{0.49\\textwidth}}
        \\centering
        \\includegraphics[width=\\textwidth]{{Pics/{ala[alaNum-1]}{fig_number}SliceP.png}}
        \\caption{{Vista lateral, colors segons la pressió.}}
        \\label{{fig:{fig_label}P}}
    \\end{{subfigure}}
    \\caption{{Vistes d'un tall lateral de l'ala {alaNum}, amb angle d'atac {degree}º.}}
    \\label{{fig:{fig_label}s}}
\\end{{figure}}
\\vspace{{15pt}}
"""

    return latex_code

# Example usage: Change the figure number as needed
for a in [0,1,2,45,90]:
    latex_output = generate_latex_figure(a,1)
    print(latex_output)
for a in [0,1,2,45,90]:
    latex_output = generate_latex_figure(a,2)
    print(latex_output)
