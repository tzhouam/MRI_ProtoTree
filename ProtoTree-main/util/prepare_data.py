import os
import shutil
from joblib import delayed,Parallel
if not os.path.isdir('/jhcnas1/zhoutaichang/prototree'):
    os.mkdir('/jhcnas1/zhoutaichang/prototree')
roots={
    '/jhcnas1/zhoutaichang/enhanced/train/':'/jhcnas1/zhoutaichang/prototree/train/'
    ,'/jhcnas1/zhoutaichang/enhanced/test/':'/jhcnas1/zhoutaichang/prototree/test/',
       '/jhcnas1/zhoutaichang/original/':'/jhcnas1/zhoutaichang/prototree/original/'}
import time
for r in roots:

    t=roots[r]
    c=roots[r]
    files = os.listdir(r)
    if not os.path.isdir(roots[r]):
        os.mkdir(roots[r])
    if not os.path.isdir(t):
        os.mkdir(t)
    if not os.path.isdir(c):
        os.mkdir(c)
    time.sleep(1)
    # for file in files:
    def f(file):
        if os.path.isdir(file):
            return
        if '_i' in file:
            shutil.copy2(r+file,c+file)
        else:
            shutil.copy2(r+file,t+file)


    Parallel(n_jobs=-1, backend='multiprocessing')(
        delayed(f)(file) for file in files)
