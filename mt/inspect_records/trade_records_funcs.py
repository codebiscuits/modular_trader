from pathlib import Path
import json
from itertools import product


def load_all(folder: Path, types: list[str]):
    """"""
    all_data = {}
    for agent, state in product(folder.glob('*'), types):
        if not agent.is_dir():
            continue
        # print(agent.parts)
        filepath = folder / agent.parts[6] / f'{state}_trades.json'

        try:
            with open(filepath, 'r') as file:
                data = json.load(file)
        except json.decoder.JSONDecodeError as e:
            data = None
        except FileNotFoundError as e:
            data = None

        if data:
            if all_data.get(agent.parts[6]):
                all_data[agent.parts[6]].update(data)
            else:
                all_data[agent.parts[6]] = data

    return all_data

def load_one(folder: Path, types: list[str]) -> list:
    """folder is the records folder for the agent in question (as a Path object), types is a list of strings
    representing the records files to be included in the output, eg ['closed', 'closed_sim'] or ['open'] etc"""

    all_data = []
    for state in types:
        filepath = folder / f'{state}_trades.json'

        try:
            with open(filepath, 'r') as file:
                data = json.load(file)
        except json.decoder.JSONDecodeError as e:
            print('decoder error')
            data = None
        except FileNotFoundError as e:
            print('file not found')
            data = None

        if len(types) == 1:
            return data

        if data:
            all_data.append(data)

    x = all_data[0]
    for d in all_data[1:]:
        x = x | d

    return x

def duration(trade):
    d = trade[-1]['timestamp'] - trade[0]['timestamp']
    return f"{int(d // 3600)}h {int(d // 60 % 60)}m {int(d % 60)}s"

