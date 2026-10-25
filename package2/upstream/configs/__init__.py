import yaml, pathlib
def paths():
    p = pathlib.Path(__file__).parent / "paths.yaml"
    return yaml.safe_load(p.read_text())