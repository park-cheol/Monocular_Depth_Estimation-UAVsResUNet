# Monocular Depth Estimation Using ResUNetPlusPlus

# Reference
- **BTS**
    - **[Paper](https://arxiv.org/abs/1907.10326)**
    - **[Github](https://github.com/cleinc/bts/tree/master/pytorch)**


- **GLPDepth**
  - **[Paper](https://arxiv.org/abs/2201.07436)**
  - **[Github](https://github.com/vinvino02/GLPDepth)**


- **ResUNet**
    - **[ResUNet Paper](https://arxiv.org/abs/1711.10684)**
    - **[ResUNet++](https://arxiv.org/abs/1911.07067)**
    - **[ResUNet Github](https://github.com/rishikksh20/ResUnet/tree/01566c9ca77184ec7ecbd21ddb0681b5941e63f4)**


- **ASPP(Atrous Spatial Pyramid Pooling)**
    - **[DeepLab V2 Paper](https://arxiv.org/abs/1606.00915)**

# Implement

- **Train(Distributed Training)**
  - **KITTI**
    > **python main.py --dist-url tcp://127.0.0.1:18214 --world-size 1 --rank 0 --multiprocessing-distributed --max-depth 80 --do-random-rotate --degree 1.0 --min-depth-eval 1e-3 --max-depth-eval 80 --max-depth 80 --eigen-crop --batch-size 8 --dataset-name kitti --garg-crop --do-kb-crop
    --input-height 352 --input-width 704**
    
  - **NYU Depth V2**
    > **python main.py --dist-url tcp://127.0.0.1:18214 --world-size 1 --rank 0 --multiprocessing-distributed --max-depth 10 --do-random-rotate --degree 2.5 --min-depth-eval 1e-3 --max-depth-eval 10 --max-depth 10 --eigen-crop --batch-size 16
    --input-height 416 --input-width 544**


    