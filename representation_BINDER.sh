#!/bin/bash
           
i=1
               
while [ $i -le 5 ];
do
       
      echo -e "\nDataset: Medical_data_direct_closure, Dimension: 128, alpha: 40, Neg_multiplier: 512" >> results/representation_full_transitive.log
      python3 . data/Medical_data_direct_closure.tsv.full_transitive 128 40 10 0 0.008 0.01 512 --reconstruction --verbose 0 --iterations 10000 --stop-width 9999 --run-full-adj yes >> results/representation_full_transitive.log
      ((i++))
	      
done

i=1
               
while [ $i -le 5 ];
do
       
      echo -e "\nDataset: Music_data_direct_closure, Dimension: 128, alpha: 25, Neg_multiplier: 128" >> results/representation_full_transitive.log
      python3 . data/Music_data_direct_closure.tsv.full_transitive 128 25 10 0 0.008 0.01 128 --reconstruction --verbose 0 --iterations 10000 --stop-width 9999 --run-full-adj yes >> results/representation_full_transitive.log
      ((i++))
	      
done

i=1
               
while [ $i -le 5 ];
do
       
      echo -e "\nDataset: animals, Dimension: 128, alpha: 25, Neg_multiplier: 512" >> results/representation_full_transitive.log
      python3 . data/animals.tsv.full_transitive 128 25 10 0 0.008 0.01 512 --reconstruction --verbose 0 --iterations 10000 --stop-width 9999 --run-full-adj yes >> results/representation_full_transitive.log
      ((i++))
	      
done

i=1
               
while [ $i -le 5 ];
do
       
      echo -e "\nDataset: Shwartz_lex_closure, Dimension: 128, alpha: 40, Neg_multiplier: 512" >> results/representation_full_transitive.log
      python3 . data/Shwartz_lex_closure.tsv.full_transitive 128 40 10 0 0.008 0.01 512 --reconstruction --verbose 0 --iterations 10000 --stop-width 9999 --run-full-adj yes >> results/representation_full_transitive.log
      ((i++))
	      
done

i=1
               
while [ $i -le 5 ];
do
       
      echo -e "\nDataset: Shwartz_random_closure, Dimension: 128, alpha: 25, Neg_multiplier: 512" >> results/representation_full_transitive.log
      python3 . data/Shwartz_random_closure.tsv.full_transitive 128 25 10 0 0.008 0.01 512 --reconstruction --verbose 0 --iterations 10000 --stop-width 9999 --run-full-adj yes >> results/representation_full_transitive.log
      ((i++))
	      
done

        
i=1
             
while [ $i -le 5 ];     
do
	      
      echo -e "\nDataset: Noun_data, Dimension: 128, alpha: 100, Neg_multiplier: 256" >> ArXiV_results/representation_full_transitive.log
      python3 . data/noun_closure.tsv.full_transitive 128 $alpha 10 0 0.008 0.01 $neg_mul --reconstruction --verbose 0 --iterations 500 --stop-width 9999 --run-full-adj yes >> results/representation_full_transitive.log
      ((i++))
	      
done
    	
