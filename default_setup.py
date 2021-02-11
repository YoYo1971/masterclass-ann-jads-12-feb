# load some default Python modules
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
#% matplotlib inline
plt.style.use('seaborn-whitegrid')

# suppress warnings
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter("ignore", DeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# add datatools directory to path
import sys
sys.path.append('../../../')

# load tensorflow
import tensorflow as tf

print(f'Loaded numpy {np.__version__} as np.')
print(f'Loaded pandas {pd.__version__} as pd')
print(f'Loaded matplotlib.pyplot {mpl.__version__} as plt')
print(f'Loaded seaborn {sns.__version__} as sns')
print(f'Loaded tensorflow as tf {tf.__version__}.')