import numpy as np
from matplotlib import pyplot as plt
import pdb
def plot(accuracy_stats,file):
    cm = plt.get_cmap('Accent')
    cm1 = plt.get_cmap('tab20b')
    colors= [cm1(0),cm1(1),cm1(2),cm1(3),cm(0),cm(2)]
    # pdb.set_trace()
    ticks = np.arange(len(accuracy_stats["bleu4"]))
    group_num = len(accuracy_stats.keys())-2

    group_width = 0.8

    bar_span = group_width / group_num

    bar_width = bar_span

    baseline_x = ticks - (group_width - bar_span) / 2
    # pdb.set_trace()
    accuracy_stats = dict(sorted(accuracy_stats.items(),key = lambda x:x[0]))
    i = 0
    if str(1) in file:
      plt.figure(figsize=(9, 4))
    else:
      plt.figure(figsize=(12, 4))
    for index, (key,y) in enumerate(accuracy_stats.items()):
        y = dict(sorted(y.items(),key = lambda x:x[0]))
        if index in [4,6]:
            # pdb.set_trace()
            continue
        plt.bar(baseline_x + i*bar_span, y.values(), bar_width,label = key,color=colors[i])
        i += 1
    plt.ylabel('Metrics',fontsize = 14)
    plt.ylim((0,0.76))
    plt.title(f'{file}',fontsize = 16)

    # pdb.set_trace()
    plt.xticks(ticks, accuracy_stats["bleu4"].keys())
    plt.xlabel('Length of Context Turns',fontsize = 14)
    plt.legend()
    plt.savefig(f'{file}.jpg',dpi=600,bbox_inches='tight')
    plt.show()

if __name__=='__main__':
accuracy_stats = np.load('acc.npy',allow_pickle=True).tolist()
plot(accuracy_stats,'1-Role Last Turn-Seg in PC')