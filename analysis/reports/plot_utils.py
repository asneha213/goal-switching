
def draw_significance_marker(ax, x1, x2, y, h, text_str, color='black'):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=color)
    ax.text((x1 + x2) * .5, y + h, text_str, ha='center', va='bottom', color=color, fontweight='bold')

def add_text_annotation(ax, text_str, x, y, fontsize=11):
    ax.annotate(text_str, xy=(x, y),
                    xycoords='axes fraction',
                    fontsize=fontsize, fontweight='bold', ha='center')