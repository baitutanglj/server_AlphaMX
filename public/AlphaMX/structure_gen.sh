#!/bin/bash
source activate alphamx
AlphaMX_dir=$(cd "$(dirname "$0")";pwd)
input=$1
file_type=$2
output=$3
python structure_gen.py $input $file_type $output