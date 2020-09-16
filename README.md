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
import imagenet.core as imagenet

dataset_dir = '/path/to/your/ILSVRC2012-dataset'
dataset = imagenet.ImageNet(dataset_dir)

trainset, validset = dataset.from_generator(
    image_size=224,
    batch_size=32,
    shuffle=True,
    take=None
)
~~~

### Looking up label

~~~python
import imagenet.core as imagenet
dataset = imagenet.ImageNet(dataset_dir='/path/to/your/ILSVRC2012-dataset')

label_index = 100

dataset.lookup[label_index]
>>> {'syn': 'n02489166', 'text': 'proboscis_monkey'}

dataset.lookup['n01632458']
>>> {'id': 497, 'text': 'spotted_salamander'}
~~~