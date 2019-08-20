import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import moxing as mox
mox.file.shift('os', 'mox')

def curve_draw(train_loss_list, train_accurate_list, test_accurate_list, log_dir, version):
    x1 = range(1, len(train_loss_list)+1)
    x2 = range(1, len(train_accurate_list)+1)
    x3 = range(1, len(test_accurate_list)+1)
    y1 = train_loss_list
    y2 = train_accurate_list
    y3 = test_accurate_list
    plt.subplot(3, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.xlabel('epochs')
    plt.subplot(3, 1, 2)
    plt.plot(x2, y2, '*-')
    plt.xlabel('epochs')
    plt.subplot(3, 1, 3)
    plt.plot(x3, y3, 'x-')
    plt.xlabel('epochs')
    #plt.savefig(log_dir + 'result' + version + '.png')
    train_loss_list_path = log_dir + "train_loss" + version + '.csv'
    train_accurate_list_path = log_dir + "train_accurate" + version + '.csv'
    test_accurate_list_path = log_dir + "test_accurate" + version + '.csv'
    train_loss_list_csv = pd.DataFrame(train_loss_list)
    train_accurate_list_csv = pd.DataFrame(train_accurate_list)
    test_accurate_list_csv = pd.DataFrame(test_accurate_list)

    with mox.file.File(train_loss_list_path,"w") as c1:
        train_loss_list_csv.to_csv(c1)
    with mox.file.File(train_accurate_list_path,"w") as c2:
        train_accurate_list_csv.to_csv(c2)
    with mox.file.File(test_accurate_list_path,"w") as c3:
        test_accurate_list_csv.to_csv(c3)
