import os, glob

jobs = glob.glob('output/*')

# sort the list of jobs based on date and time
jobs.sort(key= lambda x: (x.split("_")[-2], x.split("_")[-1]))

for idx, job in enumerate(jobs):
	print("[{}]: {}".format(idx, job.split("/")[-1]))

selected_idx = input("Select job by number: ")
selected_job = jobs[int(selected_idx)]

job_path = os.path.join(selected_job, "graph")
os.system("tensorboard --logdir {}".format(job_path))