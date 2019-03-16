import subprocess, os

command = 'gcloud compute ssh tensorflow-gpu --zone europe-west1-b --command "ls cnn_battery/output"'
result = subprocess.getoutput(command)

jobs = result.split("\n")

# sort the list of jobs based on date and time
jobs.sort(key= lambda x: (x.split("_")[-2], x.split("_")[-1]))

for idx, job in enumerate(jobs):
	print("[{}]: {}".format(idx, job))

selected_idx = input("Select job by number: ")
selected_job = jobs[int(selected_idx)]

command = 'gcloud compute scp --recurse tensorflow-gpu:~/cnn_battery/output/{} output'.format(selected_job)

os.system(command)