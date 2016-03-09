import numpy as np
from scipy.stats import pearsonr
import config

def correlate(described):
  correlation = {}
  
  correlation['std(mfcc)-Inharmonicity'] = pearsonr(described['mfcc-Std'],described['Inharmonicity'])
  correlation['Var(mfcc)-Inharmonicity'] = pearsonr(described['mfcc-Var'],described['Inharmonicity'])
  correlation['Skew(mfcc)-Inharmonicity'] = pearsonr(described['mfcc-Skew'],described['Inharmonicity'])
  correlation['Kurt(mfcc)-Inharmonicity'] = pearsonr(described['mfcc-Kurt'],described['Inharmonicity'])
  
  for iii in range(len(described['mfcc'])):
    correlation['mfcc[' + str(iii) + ']-Inharmonicity'] = pearsonr(described['mfcc'][iii],described['Inharmonicity'])
    for jjj in range(len(described['mfcc'])):
      if iii == jjj : continue
      correlation['mfcc[' + str(iii) + ']-mfcc[' + str(jjj) + ']'] = pearsonr(described['mfcc'][iii],described['mfcc'][jjj])
    
  for keys in correlation.keys():
    if abs(correlation[keys][0]) < 0.8:
      del correlation[keys]
    else:
      correlation[keys] = np.array(correlation[keys]).tolist()
  

  return correlation
