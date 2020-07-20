## To train:
<code>\
python -m trainer.train </br>
&nbsp; --model-name simple </br>
&nbsp; --job-name test </br>
&nbsp; --filters 16_32 </br>
&nbsp; --kernels 8_8 </br>
&nbsp; --feature-name boundary_edge_surface </br>
&nbsp; --path-tfrecords data/processed/boundary_edge_surface_1.0/tfrecords </br>
&nbsp; --n-train-samples 40 </br>
&nbsp; --n-test-samples 10 </br>
</code>
