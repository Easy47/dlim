import glob

def collect_INRIA_Holidays_paths(path):
    path = path.strip()
    imgs = None
    if (path[-1] == "/"):
        imgs = glob.glob(path + "*.jpg", recursive=True)
    else:
        imgs = glob.glob(path + "/*.jpg", recursive=True)
    return imgs
