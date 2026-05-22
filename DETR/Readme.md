# DETR: End-to-End Object Detection with Transformers

## Implementation Details

![alt text](images/image.png)

### Feature extractor:

![alt text](images/image-1.png)

### Transformer Encoder and Decoder:

![alt text](images/image-2.png)

![alt text](images/image-3.png)

![alt text](images/image-7.png)

### Decoding Head

![alt text](images/image-4.png)

## Losses

### Data association
The hugarian algorithms is used to associate the pred and GT by minmizing the distance betwee them.

![alt text](images/image-5.png)

## Total Loss

Classification loss: Cross entropy loss
Regression loss: L1 loss + GIoU loss

![alt text](images/image-6.png)






