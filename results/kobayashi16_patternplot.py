import matplotlib.pyplot as plt
import numpy as np
import icdlplot

if __name__ == '__main__':
    categories = ("Two Stationary",
                  "Stationary IA",
                  "Stationary IA\nIL moving backward",
                  "Stationary IL",
                  "Stationary IL\nIA moving backward",
                  "No Stationary")
    
    age9 = (16.46, 18.60, 0, 1.83, 30.79, 32.32)
    older = (34.60, 40.90, 10.20, 8.70, 0, 5.5)
    age6 = (0, 97.04, 0, 0, 0, 2.96)
    younger = (62.80, 32.60, 0, 1.2, 0, 3.5)
    
    x = np.arange(len(categories))
    width=0.33
    
    #fig, ax = plt.subplots(layout='constrained')

    #rects = ax.bar(x, age9, width, label="Age 9")
    #ax.bar_label(rects, padding=3)
    #rects = ax.bar(x+width, older, width, label="Older")
    #ax.bar_label(rects, padding=3)

    #ax.set_ylabel("%")
    #ax.set_xticks(x+width/2.0, categories)
    #ax.legend(loc='upper left', ncols=3)
    #ax.set_ylim(0, 100)

    #plt.show()

    def autopct_filter(pct):
        return ('%1.1f%%' % pct) if pct > 6 else ''

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(8, 3))
    colors = ["#99ff99",
              "#ff9999",
              "#9999ff",
              "#9f9f9f",
              "#0FEFEB",
              "#DCEB12"]
    wedges, texts, autotexts = ax1.pie(age9,
                                       labels=None,
                                       autopct=autopct_filter,
                                       colors=colors,
                                       startangle=90)
    ax2.pie(older,
            labels=None,
            autopct=autopct_filter,
            colors=colors,
            startangle=90)
    
    ax3.pie(age6,
            labels=None,
            autopct=autopct_filter,
            colors=colors,
            startangle=90)
    
    ax4.pie(younger,
            labels=None,
            autopct=autopct_filter,
            colors=colors,
            startangle=90)
    
    ax1.set_title('MIMo 9 Months')
    ax2.set_title('Kobayashi Older Infants')
    ax3.set_title('MIMo 6 Months')
    ax4.set_title('Kobayashi Younger Infants')

    legend = fig.legend(wedges,
               categories,
               loc="lower center",
               bbox_to_anchor=(0.5, 0.05),
               ncol=len(categories),
               frameon=False,
               handlelength=1,
               handleheight=1,
               borderaxespad=0,
               columnspacing=0.5)
    
    for handle in legend.legend_handles:
        handle.set_edgecolor('black')
        handle.set_linewidth(0.5)

    plt.tight_layout()
    plt.savefig('kobayashi_patterns.pdf',
                dpi=300,
                bbox_inches='tight',
                format='pdf')