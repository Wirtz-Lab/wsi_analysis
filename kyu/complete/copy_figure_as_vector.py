import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.cm import get_cmap
import pywintypes
import addcopyfighandler
import matplotlib as mpl
mpl.rcParams['savefig.format'] = 'svg'

a=[1,1,1,1,1,3,3,2,2,2,6,6,6,6,6,6,6,6,6,6,4,4,4,5,5,5,5,5,5,5,5,5]
plt.hist(a)
plt.show()


