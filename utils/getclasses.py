def getclass(yClass):
  classes = []
  
  for clas in yClass:
    if clas not in classes:
      classes.append(clas)
  return classes
