

file = '/home/khan_mo/kratos/thesis/important Git Lib/ast/src/predict/predictCSV/eventfiles/overlap_133-0020_201031_231556_indoors_ts30.txt'

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import matplotlib.dates as dt
import numpy as np

def hbarplot():
    times = [timedelta(0, 737),
             timedelta(0, 110),
             timedelta(0, 356),
             timedelta(0, 171),
             timedelta(0, 306)]

    start_date = datetime(1900, 1, 1, 0, 0, 0)
    times_datetime = [start_date + times[i] for i in range(len(times))]
    # pandas requires numerical data on dependent axis
    times_num = dt.date2num(times_datetime)
    # to make times_num proportionally correct
    for i in range(len(times_num)):
        times_num[i] -= dt.date2num(start_date)
        pass

    df = pd.DataFrame([times_num], index=['Classes'])
    fig, ax1 = plt.subplots(1, 1, sharex=True, sharey=True)
    df.plot(kind='barh', ax=ax1, stacked=True)
    plt.show()



def someplot():
    ts = pd.Series(np.random.randn(20), index=pd.date_range("1/1/2000", periods=20))

    ts = ts.cumsum()
    fig, ax1 = plt.subplots(1, 1, sharex=True, sharey=True)
    ts.plot(kind='barh', ax=ax1, stacked=True)
    plt.show()


def freq():
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np

    dates = pd.date_range(start='2020-02-14 20:30', end='2020-02-24', freq='10min')
    data = pd.DataFrame({'Timestamp': dates,
                         'Validity': (np.round(np.random.uniform(0, .02, len(dates)).cumsum()) % 2).astype(bool)})
    color = 'dodgerblue'
    plt.barh(y=1, left=data['Timestamp'], width=1/24/6, height=0.3,
             color=['none' if not val else color for val in data['Validity']])
    plt.axhline(1, color=color)

    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    # plt.xticks(rotation=30)
    plt.margins(y=0.4)
    plt.ylabel('Validity')
    plt.xlabel('Timestamp')
    plt.tight_layout()
    plt.show()

hbarplot()