import argparse
import os
import json
import pandas as pd

def extract_conf(path):
    components = path.split('/')

    # Take the last component
    last_component = components[-3]

    # Split the last component by "-"
    subcomponents = last_component.split('-')

    # Join the subcomponents except the first one
    result = "=".join(subcomponents[3:])

    return result

def extract_val(epoch,root,split,metric):
    stats_json_path = os.path.join(root, '..', '..', '41', split, 'stats.json')
    if os.path.exists(stats_json_path):
        with open(stats_json_path, 'r') as stats_json_file:
            print(extract_conf(root))
            for line in stats_json_file:
                stats_entry = json.loads(line.strip())
                if stats_entry['epoch'] == epoch:
                    return stats_entry[metric]

def agg_batch(dir,metric):
    df = pd.DataFrame(columns=["conf",f"train_{metric}",f"val_{metric}",f"test_{metric}"])
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name == 'best.json' and ("val" in root):
                best_json_path = os.path.join(root, name)
                with open(best_json_path, 'r') as best_json_file:
                    best_json = json.load(best_json_file)
                    epoch = best_json['epoch']
                    metric_val = best_json[metric]
                    # Now we need to find the stats.json file
                    metric_test = extract_val(epoch,root,"test",metric)
                    metric_train = extract_val(epoch,root,"train",metric)
                    conf = extract_conf(root)
                    entry = pd.DataFrame.from_dict({"conf":[conf],f"train_{metric}":[metric_train],f"val_{metric}":[metric_val],f"test_{metric}":[metric_test]})
                    df = pd.concat([df,entry],ignore_index=True)
    df = df.sort_values(by=f"val_{metric}",ascending=(metric=="mae"))
    df.to_csv(os.path.join(dir,"aggregated.csv"))
                    


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(
        description='Train a classification model')
    parser.add_argument('--dir', dest='dir', help='Dir for batch of results',
                        required=True, type=str)
    parser.add_argument('--metric', dest='metric',
                        help='metric to select best epoch', required=False,
                        type=str, default='auto')
    return parser.parse_args()


args = parse_args()
agg_batch(args.dir, args.metric)
