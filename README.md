# disentangle oversampling model
## data and code file
### data preparation: mnist, fashion-mnist, celebA;
###  "pre_train.py" 
- used to pre-train a generator and classifier, 
 the classifier use confidence estimation training, 
 and the generator use beta-tcvae as the based model; 
### the setting parameter is in "main.py" and "pre_train.py";
### note: 
- you need to initial the dataset in the dataset folder first, use
```python
    python dataset.py
```
### runing procedure:
- first run
```python
    python pre_train.py
```
- then run
```python
    python main.py
```