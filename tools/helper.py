import os
import csv

def make_csv_paths(apply_lora, task_name, lora_r, linear_wd):
    if apply_lora:
        loss_csv = f"test/csv/{task_name}/loss_{lora_r}_wd{linear_wd}.csv"
        accu_csv = f"test/csv/{task_name}/accu_{lora_r}_wd{linear_wd}.csv"
    else:
        loss_csv = f"test/csv/{task_name}/loss_full.csv"
        accu_csv = f"test/csv/{task_name}/accu_full.csv"

    loss_csv_dir = os.path.dirname(loss_csv)
    accu_csv_dir = os.path.dirname(accu_csv)

    if not os.path.exists(loss_csv_dir):
        os.makedirs(loss_csv_dir)
        print(f"Directory created: {loss_csv_dir}")
    if not os.path.exists(accu_csv_dir):
        os.makedirs(accu_csv_dir)
        print(f"Directory created: {accu_csv_dir}")

    with open(loss_csv, mode='w', newline='') as loss_file:
        loss_writer = csv.writer(loss_file)
        loss_writer.writerow(["epoch", "loss"])

    with open(accu_csv, mode='w', newline='') as accu_file:
        accu_writer = csv.writer(accu_file)
        accu_writer.writerow(["epoch", "accuracy"])

    return loss_csv, accu_csv