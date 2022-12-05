import csv

import argparse
import numpy as np
def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default='test')
    return parser.parse_args()

#load five csvs and calculate means and stds

def main(args):
    number_runs = 3
    metrics = ['MSE', 'RMSE', 'PSNR', 'MAE', 'SSIM', 'Mass_violation', 'Mean bias', 'MS SSIM', 'Pearson corr', 'CRPS', 'Mean abs bias', 'neg mean', 'neg num']
    dict_lists = {}
    means = {}
    for metric in metrics:
        means[metric] = []
    for i in range(number_runs):#
        filename = './data/score_log/'+ args.model_id + '_' + str(i) +'_test.csv'
        with open(filename,'r') as data:
            for line in csv.reader(data):
                if line[0] in metrics:
                    print(type(line[1]), line[1], line[0])
                    means[line[0]].append(float(line[1]))
    
    #iterate over dict lists
    for metric, values in means.items():
        print(metric, values)
        dict_lists[metric+'_mean'] = np.mean(np.array(values))
        dict_lists[metric+'_std'] = np.std(np.array(values))
        
    #save mean+std dict as csv
    save_dict(dict_lists, args)

    
                                            
def save_dict(dictionary, args):
    w = csv.writer(open('./data/score_log/'+args.model_id+'_means.csv', 'w'))
    # loop over dictionary keys and values
    for key, val in dictionary.items():
        # write every key and value to file
        w.writerow([key, val])
        

if __name__ == '__main__':
    args = add_arguments()
    main(args)