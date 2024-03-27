#!/bin/bash

for TC in 0 10
do

    for dataset in Medical_data_direct_closure Music_data_direct_closure animals Shwartz_lex_closure Shwartz_random_closure
    do
                
            i=1
               
            while [ $i -le 5 ];
            do
       
	          echo -e "\nDataset: $dataset, TC: $TC, Dimension: 128, alpha:25000, Neg_multiplier: 128" >> results/transitive_closure.log
	          python3 . data/${dataset}.tsv.full_transitive --train data/${dataset}.tsv.train_${TC}percent --val data/${dataset}.tsv.valid --val-neg data/${dataset}.tsv.valid_neg --test data/${dataset}.tsv.test --test-neg data/${dataset}.tsv.test_neg 128 25000 10 0 0.008 0.01 128 --verbose 0 --iterations 10000 --stop-width 9999 >> results/transitive_closure.log
	          ((i++))
	      
	    done
	   
    done
        
done


for TC in 25 50
do

    for dataset in Medical_data_direct_closure Music_data_direct_closure animals Shwartz_lex_closure Shwartz_random_closure
    do
                
            i=1
               
            while [ $i -le 5 ];
            do
       
	          echo -e "\nDataset: $dataset, TC: $TC, Dimension: 128, alpha:50000, Neg_multiplier: 128" >> results/transitive_closure.log
	          python3 . data/${dataset}.tsv.full_transitive --train data/${dataset}.tsv.train_${TC}percent --val data/${dataset}.tsv.valid --val-neg data/${dataset}.tsv.valid_neg --test data/${dataset}.tsv.test --test-neg data/${dataset}.tsv.test_neg 128 50000 10 0 0.008 0.01 128 --verbose 0 --iterations 10000 --stop-width 9999 >> results/transitive_closure.log
	          ((i++))
	      
	    done
	   
    done
        
done


#noun

for TC in 0 10
do
           
    i=1
               
    while [ $i -le 5 ];
    do
       
	    echo -e "\nDataset: noun_closure, TC: $TC, Dimension: 128, alpha: 25000, Neg_multiplier: 128" >> results/transitive_closure.log
	    python3 . data/noun_closure.tsv.full_transitive --train data/noun_closure.tsv.train_${TC}percent --val data/noun_closure.tsv.valid --val-neg data/noun_closure.tsv.valid_neg --test data/noun_closure.tsv.test --test-neg data/noun_closure.tsv.test_neg 128 25000 10 0 0.008 0.01 128 --verbose 0 --iterations 10000 --stop-width 9999 >> results/transitive_closure.log
	    ((i++))
	      
    done
	   
done
        
for TC in 25 50
do
           
    i=1
               
    while [ $i -le 5 ];
    do
       
	    echo -e "\nDataset: noun_closure, TC: $TC, Dimension: 128, alpha: 50000, Neg_multiplier: 12" >> results/transitive_closure.log
	    python3 . data/noun_closure.tsv.full_transitive --train data/noun_closure.tsv.train_${TC}percent --val data/noun_closure.tsv.valid --val-neg data/noun_closure.tsv.valid_neg --test data/noun_closure.tsv.test --test-neg data/noun_closure.tsv.test_neg 128 50000 10 0 0.008 0.01 12 --verbose 0 --iterations 10000 --stop-width 9999 >> results/transitive_closure.log
	    ((i++))
	      
    done
	   
done
