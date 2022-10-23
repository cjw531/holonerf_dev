import os
import csv
import configargparse
import random

def config_parser():    
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--refdir", type=str, help='reference original base image directory')
    parser.add_argument("--savedir", type=str, help='reduced image save directory')
    parser.add_argument("--factor", type=float, default=1.0, help='training set = factor * training set')
    return parser

def select_image():
    parser = config_parser()
    args = parser.parse_args()

    original_files = sorted(os.listdir(args.refdir + '/images/')) # original images filename path list
    select_num = int(len(original_files) * args.factor) # number of samples
    selected_indices = random.sample(range(0, len(original_files)), select_num) # randomly select images

    os.system("mkdir " + args.savedir)
    os.system("mkdir " + args.savedir + '/images')

    csv_data = open(args.savedir + '/hidden_set.csv', 'w')
    csv_writer = csv.writer(csv_data)
    csv_writer.writerow(['index', 'filename', 'set'])
    
    for i in range(len(original_files)):
        if i not in selected_indices: csv_writer.writerow([i, args.refdir + '/images/' + original_files[i], 'hidden']) # save hidden.csv to the new img directory
        else: os.system('cp ' + args.refdir + '/images/' + original_files[i] + ' ' + args.savedir + '/images/') # copy image to new directory

    csv_data.close()

if __name__ == '__main__':
    select_image()