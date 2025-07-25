import json

def jload(filepath):
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def jsave(data, filepath):
    """Save data to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2) 