# CSIC code for Continual Learning for Sentiment Classification by Iterative Networks Combination

### DATA
You need to download these data and set them into data/ .
| Sentiment Classification | [16 tasks sentiment](https://drive.google.com/file/d/1lgT2ieGn5sAXwtF_nFH4ee0ZmNC_hFnY/view?usp=sharing) |

### you can run CSIC model use:
```bash
$ python main.py --logname test --seed 1 --lambda_1 0.3 --lambda_2 1e-2
```
-`--logname`: the log name which save in result_data/csvdata.
-`--seed`: we use 1,2,3 in our experiment
-`--lambda_1` `--lambda_2`: loss function hyper-parameters



### Requirements

- Python 3.7
- Pytorch 1.6.0+cu10.1 / CUDA 10.1
- transformers


Reference

BERT Base network is from https://github.com/huggingface/transformers
