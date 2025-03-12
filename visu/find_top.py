import os
import re
import pandas as pd

def find_log_files(result_dir):
    log_files = []
    for root, dirs, files in os.walk(result_dir):
        for file in files:
            if file == 'log.txt':
                log_files.append(os.path.join(root, file))
    return log_files

def extract_metrics_from_line(line):
    pattern = r'acc=([0-9.]+) pre=([0-9.]+) sen=([0-9.]+) f1=([0-9.]+) spec=([0-9.]+) kappa=([0-9.]+) my_auc=([0-9.]+) qwk=([0-9.]+)'
    match = re.search(pattern, line)
    if match:
        return {
            'acc': float(match.group(1)),
            'pre': float(match.group(2)),
            'sen': float(match.group(3)),
            'f1': float(match.group(4)),
            'spec': float(match.group(5)),
            'kappa': float(match.group(6)),
            'my_auc': float(match.group(7)),
            'qwk': float(match.group(8))
        }
    return None

def get_metrics(log_file):
    with open(log_file, 'r') as file:
        lines = file.readlines()
        if len(lines) < 2:
            return None
        return extract_metrics_from_line(lines[-2])

def save_metrics_to_csv(metrics_list, output_csv, top):
    df = pd.DataFrame(metrics_list)
    df = df.sort_values(by='acc', ascending=False).head(top)
    df.to_csv(output_csv, index=False)
    return df

def main(result_dir, output_csv, top):
    log_files = find_log_files(result_dir)
    metrics_list = []

    for log_file in log_files:
        metrics = get_metrics(log_file)
        if metrics is not None:
            metrics['file'] = log_file
            metrics_list.append(metrics)

    return save_metrics_to_csv(metrics_list, output_csv, top)

if __name__ == '__main__':
    result_dir = '/home/ubuntu/qujunlong/txyy/output/results'
    output_csv = 'top_metrics.csv'
    top = 50
    df = main(result_dir, output_csv, top)
    print(df)