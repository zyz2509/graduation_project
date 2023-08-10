# what I try to do?
A model for 3D object detection with encoder-decoder structure.

# How I prepare to do it?
Inspired by PointNet/PointNet++ and PointTransformer v2,I designed a hierarchic backbone(named after 'SA' and 'FP' just like those in the original pointnet) and have a progressive receptive field.I'm trying to use the point method in a voxel manner,which takes full advantage of the high efficiency and the bottom-up structure of the point-based method,and trades off some quantification accuracy and memory consumption for a faster behavior.

# Details of my work?
Backbone:
    The backbone contains 3(+) Set Abstraction(SA) layers.It doesn't need FPS and BallQuery because of the voxel processing.And I'm thinking whether to use pointnet or local attention to process features of each voxel.
    Additionally,the decoder for object detection is 3(+) Feature Propagation(FP) layers.Instead of the widely used interpolation method,I tend to use the modified hough voting method, just like the one used in VoteNet.Specifically,in FP layer,vote is applied to every voxel to obtain the offset of the corresponding spatial coordinate(it tends to be the already shifted coordinates in recent works),and the same applies to features.In the training process,an additional loss function is used to ensure the shifted coordinates are restricted in the neighborhood of the object center.

Neck&Head:to be done...
