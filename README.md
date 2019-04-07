# IoM-IoU

## Online real-time high-quality action tube construction for spatiotemporal action localization

In this paper, we study the challenging problem of constructing high-quality action tubes in video with multiple overlapping action instances. Most existing approaches only link the current bounding box to the action tubes when the intersection over union (IoU) between the current and last bounding boxes of the action tube exceeds a threshold. We propose a novel action tube construction approach for online real-time multiple spatiotemporal action localization and classification. Compared with online and offline methods, our approach achieves state-of-the-art performance for both appearance and optical flow data on the UCF-101-24 untrimmed dataset. Major contributions can be summarized as follows:
(1) We propose a novel IoU computation approach (IoM-IoU) to overcome the shortcomings in handling action instance overlap.
(2) We take both IoM-IoU computation and score reweighting into consideration to yield robust reweighted scores for constructing action tubes in an online fashion.
(3) Our action tube construction approach can be inserted into existing convolutional architectures for action localization/classification in real time.

![这里随便写文字](https://github.com/djzgroup/IoM-IoU/blob/master/pipeline.jpg)


## Performance
Action localization results (mAP) on untrimmed videos of the UCF101-24 dataset in split1.
<table style="width:100% th">
  <tr>
    <td>IoU Threshold = </td>
    <td>mAP@0.20</td> 
    <td>mAP@0.50</td>
    <td>mAP@0.75</td>
    <td>mAP@0.5:0.95</td>
  </tr>
  <tr>
    <td align="left">Yu et al. [1] </td> 
    <td>26.5</td>
    <td>--</td>
    <td>--</td> 
    <td>--</td>
  </tr>
  <tr>
    <td align="left">Weinzaepfel et al. [2] </td> 
    <td>46.8</td>
    <td>--</td> 
    <td>--</td>
    <td>--</td>
  </tr>
  <tr>
    <td align="left">Peng et al. [3] </td> 
    <td>73.5</td>
    <td>32.1</td> 
    <td>2.7</td>
    <td>7.3</td>
  </tr>
  <tr>
    <td align="left">Singh et al [4] </td> 
    <td>66.6</td>
    <td>36.4</td>
    <td>7.9</td> 
    <td>14.4</td> 
  </tr>
  <tr>
    <td align="left">He et al. [5] w/o LSTM</td> 
    <td>70.3</td>
    <td>--</td>
    <td>--</td>
    <td>--</td>
  </tr>
  <tr>
    <td align="left">He et al. [5] with LSTM</td> 
    <td>71.7</td>
    <td>--</td>
    <td>--</td>
    <td>--</td>
  </tr>
    <tr>
    <td align="left">Singh et al. [6] Appearance</td> 
    <td>69.8</td>
    <td>40.9</td>
    <td>15.5</td>
    <td>18.7</td>
  </tr>
  <tr>
    <td align="left">Ours-Appearance</td> 
    <td>72.5</td>
    <td>41.8</td>
    <td>15.2</td>
    <td>18.7</td>       
  </tr>
    </tr>
  <tr>
    <td align="left">Singh et al. [6] Accurate-Flow</td> 
    <td>63.7</td>
    <td>30.8</td>
    <td>2.8</td>
    <td>11.0</td>
  </tr>
  <tr>
    <td align="left">Ours-Accurate-Flow</td> 
    <td>67.4</td>
    <td>32.7</td>
    <td>3.7</td>
    <td>11.8</td>       
  </tr>
    <tr>
    <td align="left">Singh et al. [6] Real-Time-Flow</td> 
    <td>42.5</td>
    <td>13.9</td>
    <td>0.5</td>
    <td>3.3</td>
  </tr>
  <tr>
    <td align="left">Ours-Real-Time-Flow</td> 
    <td>42.7</td>
    <td>15.3</td>
    <td>0.4</td>
    <td>3.8</td>       
  </tr>
     
</table>

## CODE
The code is developed in the TensorFlow environment.
```bash
You can verify the method of this article by executing “run_OUR.m”.
```

## References
- [1] Gang Yu and Junsong Yuan. Fast action proposals for human action detection and search. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1302-1311, 2015.
- [2] Philippe Weinzaepfel, Zaid Harchaoui, and Cordelia Schmid. Learning to track for spatio-temporal action localization. In Proceedings of the IEEE international conference on computer vision, pages 3164-3172, 2015.
- [3] Xiaojiang Peng and Cordelia Schmid. Multi-region two-stream r-cnn for action detection. In European Conference on Computer Vision, pages 744-759. Springer, 2016.
- [4] Suman Saha, Gurkirt Singh, Michael Sapienza, Philip HS Torr, and Fabio Cuzzolin. Deep learning for detecting multiple space-time action tubes in videos. British Machine Vision Conference, 2016.
- [5] Jiawei He, Mostafa S Ibrahim, Zhiwei Deng, and Greg Mori. Generic tubelet proposals for action localization. arXiv preprint arXiv:1705.10861, 2017.
- [6] gurkirt Singh, Suman Saha, Michael Sapienza, Philip Torr, and Fabio Cuzzolin. Online real-time multiple spatiotemporal action localisation and prediction. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 3637-3646, 2017.

## Acknowledgment
This work was supported in part by the National Natural Science Foundation of China under Grant 61702350 and Grant 61472289 and in part by the Open Project Program of the State Key Laboratory of Digital Manufacturing Equipment and Technology, HUST, under Grant DMETKF2017016.
