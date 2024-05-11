# Usage

```python
from utils import CudaDataLoader
from utils import MultiEpochsDataLoader


Myloader = MultiEpochsDataLoader(data_set, batch_size=128, 
                               shuffle=True, num_workers=32, 
                               pin_memory=True)
if torch.cuda.is_available():
    Myloader = CudaDataLoader(Myloader, 'cuda')

for image in Myloader:
    train_model(image)
    ...
```

