## To train:
<code>\
python -m trainer.train \
&nbsp; --model-name simple \
&nbsp; --job-name test \
&nbsp; --filters 16_32 \
&nbsp; --kernels 8_8 \
&nbsp; --feature-name boundary_edge_surface \
&nbsp; --path-tfrecords data/processed/boundary_edge_surface_1.0/tfrecords \
&nbsp; --n-train-samples 40 \
&nbsp; --n-test-samples 10\
</code>
