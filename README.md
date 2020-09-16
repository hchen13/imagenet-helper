# ImageNet Helper

Helper tools to manage ImageNet ILSVRC 2012 dataset and generate training image batches. 

## 1. Installation

Run the `setup.py` file with the following command to install the package:
~~~bash
python setup.py install
~~~
*Make sure running the above command in the project directory*

## 2. Usage

### Creating dataset generator
~~~python
import imagenet

dataset_dir = '/path/to/your/ILSVRC2012-dataset'
dataset = imagenet.ImageNet(dataset_dir)

trainset, validset = dataset.from_generator(
    image_size=224,
    batch_size=32,
    shuffle=True,
    take=None
)

for x, y in trainset:
    print(type(x), type(y))
    print(x.shape, y.shape)
    print(x.dtype, y.dtype)
    do_something_with_batch_data(x, y)
~~~
Expected outputs:
~~~bash
>>> <class 'tensorflow.python.framework.ops.EagerTensor'> <class 'tensorflow.python.framework.ops.EagerTensor'>
>>> (32, 224, 224, 3) (32,)
>>> <dtype: 'uint8'> <dtype: 'int32'>
~~~
The `dataset.from_generator()` function will return a tuple of 2 generators representing the **training** and 
**validation** datasets respectively. 

Usage of each generator is displayed above, notice that `y` values are label indices whose values are in [1, 1000],
as defined by [ImageNet classes](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).

### Looking up label

Sometimes one needs to convert label indices to human-readable text or look up label indices from label synset 
(which can often happen when dealing with ILSVRC training set), the package provides useful tools to implement
such functionalities.
~~~python
import imagenet


dataset = imagenet.ImageNet(dataset_dir='/path/to/your/ILSVRC2012-dataset')
label_index = 100

dataset.lookup[label_index]
>>> {'syn': 'n02489166', 'text': 'proboscis_monkey'}

dataset.lookup['n01632458']
>>> {'id': 497, 'text': 'spotted_salamander'}
~~~