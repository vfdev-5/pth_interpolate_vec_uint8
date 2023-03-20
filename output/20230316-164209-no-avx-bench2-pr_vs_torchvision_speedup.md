[---------------------------------------------------------------------------------------- Resize ---------------------------------------------------------------------------------------]
                                                                                 |  Pillow (9.0.0.post1)  |  torch (2.1.0a0+git0968a5d) PR  |  torchvision resize  |  Speed-up: PTH vs TV
1 threads: ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      3 torch.uint8 channels_last bilinear (64, 64) -> (224, 224) aa=True        |          61.1          |              1073.6             |        1146.8        |          1.1        
      3 torch.uint8 channels_last bilinear (224, 224) -> (270, 268) aa=True      |         132.6          |              1979.0             |        1984.4        |          1.0        
      3 torch.uint8 channels_last bilinear (256, 256) -> (1024, 1024) aa=True    |         950.0          |             21383.4             |       32588.6        |          1.5        
      3 torch.uint8 channels_last bilinear (224, 224) -> (64, 64) aa=True        |          52.8          |               447.1             |         475.5        |          1.1        
      3 torch.uint8 channels_last bilinear (270, 268) -> (224, 224) aa=True      |         139.7          |              1670.8             |        1654.0        |          1.0        
      3 torch.uint8 channels_last bilinear (1024, 1024) -> (256, 256) aa=True    |         694.8          |              7891.4             |        8518.8        |          1.1        
      3 torch.uint8 channels_last bilinear (64, 64) -> (224, 224) aa=False       |                        |              1039.0             |        1012.9        |          1.0        
      3 torch.uint8 channels_last bilinear (224, 224) -> (270, 268) aa=False     |                        |              1930.9             |        1513.2        |          0.8        
      3 torch.uint8 channels_last bilinear (256, 256) -> (1024, 1024) aa=False   |                        |             20693.9             |       20902.6        |          1.0        
      3 torch.uint8 channels_last bilinear (224, 224) -> (64, 64) aa=False       |                        |               273.1             |         184.5        |          0.7        
      3 torch.uint8 channels_last bilinear (270, 268) -> (224, 224) aa=False     |                        |              1564.6             |        1117.2        |          0.7        
      3 torch.uint8 channels_last bilinear (1024, 1024) -> (256, 256) aa=False   |                        |              4111.0             |        2976.7        |          0.7        
      3 torch.uint8 channels_first bilinear (64, 64) -> (224, 224) aa=True       |          61.1          |               675.7             |         685.4        |          1.0        
      3 torch.uint8 channels_first bilinear (224, 224) -> (270, 268) aa=True     |         132.6          |              1404.7             |        1302.0        |          0.9        
      3 torch.uint8 channels_first bilinear (256, 256) -> (1024, 1024) aa=True   |         962.9          |             12884.1             |       13109.0        |          1.0        
      3 torch.uint8 channels_first bilinear (224, 224) -> (64, 64) aa=True       |          52.9          |               413.8             |         436.9        |          1.1        
      3 torch.uint8 channels_first bilinear (270, 268) -> (224, 224) aa=True     |         138.7          |              1302.2             |        1191.7        |          0.9        
      3 torch.uint8 channels_first bilinear (1024, 1024) -> (256, 256) aa=True   |         698.5          |              7376.4             |        7699.3        |          1.0        
      3 torch.uint8 channels_first bilinear (64, 64) -> (224, 224) aa=False      |                        |               674.1             |        1192.0        |          1.8        
      3 torch.uint8 channels_first bilinear (224, 224) -> (270, 268) aa=False    |                        |              1403.0             |        1848.4        |          1.3        
      3 torch.uint8 channels_first bilinear (256, 256) -> (1024, 1024) aa=False  |                        |             12861.9             |       27802.8        |          2.2        
      3 torch.uint8 channels_first bilinear (224, 224) -> (64, 64) aa=False      |                        |               241.1             |         297.6        |          1.2        
      3 torch.uint8 channels_first bilinear (270, 268) -> (224, 224) aa=False    |                        |              1198.3             |        1418.6        |          1.2        
      3 torch.uint8 channels_first bilinear (1024, 1024) -> (256, 256) aa=False  |                        |              3617.5             |       13951.6        |          3.9        

Times are in microseconds (us).
