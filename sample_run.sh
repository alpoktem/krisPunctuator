
model="data/sample_model_p-meanf0.pcl"
vocabulary="data/vocabulary.pcl"
input_proscript="data/sample_proscript.pcl"
out_predictions="testout.txt"


python punctuator.py -m $model -v $vocabulary -i $input_proscript -o $out_predictions -p -f mean.f0.id