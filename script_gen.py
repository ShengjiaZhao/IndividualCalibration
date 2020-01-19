import os
import sys
import subprocess
import time

index = 20
while True:
    root_path = 'Results/Linear%d/' % index
    if not os.path.isdir(root_path):
        os.makedirs(root_path)

    # train_sizes = [int(round(1.5 ** i + 10 * i)) + 47 for i in range(5, 45)]
    # print(train_sizes)

    train_sizes = [int(round(2 ** i + 10 * i)) - 40 for i in range(5, 18)][::-1]
    print(train_sizes)

    datasets = ['mnist']
    attackers = ['acc'] # ['acc', 'simba']
    classifiers = ['logistic']

    max_thread = 12
    cur_running = []
    for train_size in train_sizes:
        for attacker in attackers:
            for classifier in classifiers:
                for dataset in datasets:
                    name = 'dataset=%s-attacker=%s-classifier=%s-train_size=%d' % (dataset, attacker, classifier, train_size)
                    if os.path.isfile(root_path + '%s.png' % name):
                        os.remove(root_path + '%s.png' % name)
                    subprocess.Popen(('python3 main.py --train_size=%d --log_path=%s --silent=0 --debug=0 --dataset=%s --attacker=%s --classifier=%s'
                                      % (train_size, root_path, dataset, attacker, classifier)).split())
                    cur_running.append(name)
                    print("Launched %s" % name)

                    while len(cur_running) >= max_thread:
                        for name in cur_running:
                            if os.path.isfile(root_path + '%s.png' % name):
                                cur_running.remove(name)
                        time.sleep(10)

    # Wait until all threads are finished
    while len(cur_running) != 0:
        for name in cur_running:
            if os.path.isfile(root_path + '%s.png' % name):
                cur_running.remove(name)
        time.sleep(10)
    index += 1

