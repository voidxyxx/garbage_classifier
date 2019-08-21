import os
import pandas as pd
import numpy as np
import moxing as mox
mox.file.shift('os', 'mox')

def curve_draw(train_loss_list, train_accurate_list, test_accurate_list, log_dir, version):
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
