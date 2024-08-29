import yaml

def get_config_dict():
    # Load the YAML file
    with open('src\config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    return config

