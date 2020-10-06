# Identification people by SVM
This source code will show that how to develop a system of identification of people. It can be used in attendance system by face.
Our backbone in this code:
1. [Face_recognition](https://github.com/ageitgey/face_recognition): This repo can get face locations and encode it.
2. SVM algorithm.

## 1. Preparing data
The data of images of peole which will be trained must be prepared as below:

- PERSON1:
  + image1.jpg
  + image2.jpg
  + ...
- PERSON2:
  + image1.jpg
  + image2.jpg
  + ...

It should be put in `data/` folder

## 2. Encoding images

`
python load_img.py --train_dir <data dir> --test_dir <test_dir> --save_train <save dir of encodings of train> --save_test <save dir of encodinges of test>
`


All encodings are stored in `.npy` files

## 3. Training

`
python train.py --model <save dir of model>
`

model will stored in `svm_model.sav` file in model dir (default: `model/`)

## 4. Run demo

`
python demo.py --model <save dir of model> --img_dir <dir of test image>
`

## 5. Test on testset and show accuracy

`
python test.py --model <save dir of model> --test_dir <dir of test images>
`


Example result:

```python
accuracy:  100.0
Predict actually: 70/70
```

