#!/bin/bash
lang='esp'

for folder in {1..10}
do
    python train.py --train_file ./$lang/train.$folder --test_file ./$lang/test.$folder --emb_file ~/scratch/embeddings/glove.twitter.27B.100d.txt --checkpoint ./checkpoint/ --caseless --high_way --use_crf --seg_loss 2 --ent_loss 2
done

f1_sum=0.0
rec_sum=0.0
pre_sum=0.0
acc_sum=0.0
while read f1 rec pre acc
do
    f1_sum=$(echo "$f1_sum + $f1" | bc -l)
    rec_sum=$(echo "$rec_sum + $rec" | bc -l)
    pre_sum=$(echo "$pre_sum + $pre" | bc -l)
    acc_sum=$(echo "$acc_sum + $acc" | bc -l)

done < cross_validation_record.txt

avg_f1=$(echo "scale=2; $f1_sum / 10.0" | bc -l)
avg_rec=$(echo "scale=2; $rec_sum / 10.0" | bc -l)
avg_pre=$(echo "scale=2; $pre_sum / 10.0" | bc -l)
avg_acc=$(echo "scale=2; $acc_sum / 10.0" | bc -l)

echo "f1" "recall" "precision" "accuracy"
echo $avg_f1 $avg_rec $avg_pre $avg_acc
rm cross_validation_record.txt
