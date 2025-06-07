import csv
import matplotlib.pyplot as plt

def rule_performance(file):
    """Obtain plot for the relationship between model performance and rule increment"""
    data = []
    with open(file, mode='r', encoding='utf-8-sig') as f:
        reader = csv.reader(f, delimiter=';')
        headers = next(reader) 
        headers[0] = 'Rule_Number' 
        for row in reader:
            first_col = int(row[0].split()[0]) if row[0].strip() else 0
            rest_cols = [float(x) for x in row[1:]]
            data.append([first_col] + rest_cols)

    # Extract columns
    rule_numbers = [row[0] for row in data]
    c_acc = [row[1] for row in data]
    cer = [row[2] for row in data]
    ver = [row[3] for row in data]
    edit_dist = [row[4] for row in data]
    feat_dist = [row[5] for row in data]

    def normalize(values):
        min_val = min(values)
        max_val = max(values)
        return [(x - min_val)/(max_val - min_val) for x in values]

    # Normalize data
    norm_c_acc = normalize(c_acc)
    norm_cer = normalize(cer)
    norm_ver = normalize(ver)
    norm_edit_dist = normalize(edit_dist)
    norm_feat_dist = normalize(feat_dist)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(rule_numbers, norm_c_acc, marker='o', label='C_ACC')
    ax1.plot(rule_numbers, norm_cer, marker='s', label='CER')
    ax1.plot(rule_numbers, norm_ver, marker='^', label='VER')
    ax1.set_ylabel('Normalized Score')
    ax1.set_title('Normalized Accuracy and Error Rates')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(rule_numbers, norm_edit_dist, marker='x', linestyle='--', label='EDIT_DIST')
    ax2.plot(rule_numbers, norm_feat_dist, marker='d', linestyle='--', label='FEAT_DIST')
    ax2.set_ylabel('Normalized Score')
    ax2.set_xlabel('Number of Rules')
    ax2.set_title('Normalized Distance Metrics')
    ax2.legend()
    ax2.grid(True)

    plt.xticks(rule_numbers)  
    plt.tight_layout()
    plt.savefig('normalized_metrics.png')
    plt.show()

if __name__=='__main__':
    link='rule_performance.csv'
    rule_performance(link)