import yaml
import argparse
parser = argparse.ArgumentParser()
with open('superparams.yaml', encoding='utf-8') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)['SST2']
    # print(config['SST2'])
    for key in config.keys():
        name = '--' + key
        parser.add_argument(name, type=type(config[key]), default=config[key])
args = parser.parse_args()
print(args)
    