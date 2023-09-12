from pathlib import Path
import json
from itertools import product


def load_all(folder: Path, types: list[str]):
    all_data = {}
    for agent, state in product(folder.glob('*'), types):
        filepath = folder / agent.parts[7] / f'{state}_trades.json'

        try:
            with open(filepath, 'r') as file:
                data = json.load(file)
        except json.decoder.JSONDecodeError as e:
            data = None
        except FileNotFoundError as e:
            data = None

        if data:
            if all_data.get(agent.parts[7]):
                all_data[agent.parts[7]].update(data)
            else:
                all_data[agent.parts[7]] = data

    return all_data

def duration(trade):
    d = trade[-1]['timestamp'] - trade[0]['timestamp']
    return f"{int(d // 3600)}h {int(d // 60 % 60)}m {int(d % 60)}s"

