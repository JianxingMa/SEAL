#! /bin/bash
for dataset in USAir NS PB Yeast Celegans Power Router Ecoli
do
cat $dataset"_res.txt" | awk '{if (NR==1) print;}'
done
for dataset in USAir NS PB Yeast Celegans Power Router Ecoli
do
cat $dataset"_res.txt" | awk '{if (NR==2) print;}'
done
for dataset in USAir NS PB Yeast Celegans Power Router Ecoli
do
cat $dataset"_res.txt" | awk '{if (NR==3) print;}'
done
