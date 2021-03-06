get_abs_filename() {
  # $1 : relative filename
  echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}

model=`get_abs_filename "models/Model_single-stage_kris-p-mf0_h100_lr0.05.pcl"`
vocabulary=`get_abs_filename "models/vocabulary.pcl"`

input_proscript=`get_abs_filename $1`
out_predictions=`get_abs_filename $2`

python punctuator.py -m $model -v $vocabulary -i $input_proscript -o $out_predictions -p -f f0_mean -r