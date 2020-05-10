import yaml
def load_config():
    with open("config.yaml", "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    imp_dir = ['code','data','results']
    for i in imp_dir:
        cfg[i+'_dir'] = cfg['base_dir'] + i.capitalize() +'/'
    return cfg
