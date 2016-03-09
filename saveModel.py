from sklearn.externals import joblib

def save(clf):
  joblib.dump(clf, 'filename.pkl')
  return 
