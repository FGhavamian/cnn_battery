rm-out:
	rm -r output/*

rm-rpt:
	rm -r report/*

preprocess:
	python3 -m trainer.preprocess --data-path $(P)

train:
	python3 -m trainer.train --data-path $(P) --model-name $(N) --learning-rate $(LR)

send-data:
	gcloud compute scp --recurse $(P) tensorflow-gpu:~/cnn_battery/data/

send-src:
	gcloud compute scp --recurse trainer/ tensorflow-gpu:~/cnn_battery/

send-mk:
	gcloud compute scp Makefile tensorflow-gpu:~/cnn_battery

get-model:
	python3 -m pipelines.get_model

tb:
	python3 -m pipelines.run_tensorboard

tr:
	python3 -m apps.test_report --data-path $(P)