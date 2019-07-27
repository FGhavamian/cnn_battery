# Common commands

## Training:
python -m \
trainer.train \
--epoch-num=5 \
--learning-rate=1e-2 \
--model-name=simple_cnn \
--path-tfrecords=data/ex2 \
--job-name=test \
--feature-name=boundary

## preprocess:
<code>python -m 
trainer.preprocess 
--path-data data/ex2</code>
