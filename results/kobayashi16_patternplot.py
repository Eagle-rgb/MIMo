import matplotlib.pyplot as plt
import numpy as np
import icdlplot

if __name__ == '__main__':
    categories_ = ("Two Stationary",
                  "Stationary IA",
                  "Stationary IA\nIL moving backward",
                  "Stationary IL",
                  "Stationary IL\nIA moving backward",
                  "No Stationary")
    
    categories = ("TS",
                  "IA",
                  "IA\nIL Back",
                  "IL",
                  "IL\nIA Back",
                  "NS")

    age9_100_250 = np.array([54.0, 61, 0, 6, 101, 106])
    age9_300_83 = np.array([161.0, 77, 0, 13, 49, 28])
    age9_300_250 = np.array([236.0, 20, 0, 0, 69, 3])
    age9_100_83 = np.array([24.0, 64, 0, 11, 69, 160])
    older = np.array([34.60, 40.90, 10.20, 8.70, 0, 5.5])  # in percent
    age6_100_83 = np.array([0, 63.0, 0, 0, 0, 72])
    age6_300_83 = np.array([28.0, 105, 0, 0, 0, 2])
    age6_300_250 = np.array([78.0, 57, 0, 0, 0, 0])
    age6_100_250 = np.array([0, 131.0, 0, 0, 0, 4])
    younger = np.array([62.80, 32.60, 0, 1.2, 0, 3.5])  # in percent
    
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
        return ('%1.1f%%' % pct) if pct > 10 else ''
    
    colors = ["#44dd44",
        "#dc6767",
        "#22aaaa",
        "#dd00dd",
        "#3377ff",
        "#dddB12"]
    
    age9 = [age9_100_250, age9_300_83, older]
    age6 = [age6_100_250, age6_300_83, younger]

    # Normalize
    for indx in range(len(age9)-1):
        cnt = sum(age9[indx])
        age9[indx] /= cnt
        age9[indx] *= 100.0

    for indx in range(len(age6)-1):
        cnt = sum(age6[indx])
        age6[indx] /= cnt
        age6[indx] *= 100.0

    fig, ax = plt.subplots(2, len(age9), figsize=(5, 3.2), gridspec_kw={'hspace': -0.2})

    for indx in range(len(age9)):
        data = age9[indx]
        if indx == 0:
            wedges, texts, autotexts = ax[0][indx].pie(data,
                labels=None,
                autopct=autopct_filter,
                colors=colors,
                startangle=90)
                
        else:
            ax[0][indx].pie(data,
                    labels=None,
                    autopct=autopct_filter,
                    colors=colors,
                    startangle=90)
            
    for indx in range(len(age6)):
        data = age6[indx]
        ax[1][indx].pie(data,
                labels=None,
                autopct=autopct_filter,
                colors=colors,
                startangle=90)
        
    
    
    #ax1.set_title('MIMo 9 Months')
    #ax2.set_title('Kobayashi Older Infants')
    #ax3.set_title('MIMo 6 Months')
    #ax4.set_title('Kobayashi Younger Infants')

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

    ax[0,0].set_title("MIMo 1")
    ax[0,1].set_title("MIMo 2")
    ax[0,2].set_title("Kobayashi")
    ax[0,0].set_ylabel("Older", rotation=90)
    ax[1,0].set_ylabel("Younger", rotation=90)

    #plt.tight_layout()
    plt.savefig('kobayashi_patterns.pdf',
                dpi=300,
                bbox_inches='tight',
                format='pdf')