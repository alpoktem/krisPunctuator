
model="data/sample_proscript.pcl"
vocabulary="data/vocabulary.pcl"
input_proscript="data/AlGore2006.0003.pcl"
out_predictions="testout.txt"


python punctuator.py -m $model -v $vocabulary -i $input_proscript -o $out_predictions -p -f mean.f0.id