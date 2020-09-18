# Assignment 5

### Final result:
    Corpus BLEU: 36.260917989579575

### How to run:
```
 sh run.sh vocab

 sh run.sh train

 sh run.sh test
```

### Tips:
 1. Do not apply view() directly on the input parameter of forward function in embedding layer, which will cause the loss to blow up in GPU mode. 

 2. Customized sanity check is important. I'm so lazy that I did them all in the REPEL (In fact I didn't do much XD).

 3. Pad and UNK charactor need to be replaced by \<pad\> and \<unk\> in order to pass some sanity check.

 4. Use torch.max or nn.functional.adaptive_max_pool1d for global max pooling.
