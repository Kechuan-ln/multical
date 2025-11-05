This is to generate various calibration svg calibration patterns.

``` bash
python gen_pattern.py --help
```


Reference: https://docs.opencv.org/4.x/da/d0d/tutorial_camera_calibration_pattern.html



## Examples

The one used to generate B1 size charuco board

```bash
python gen_pattern.py -o charuco_board.svg --rows 14 --columns 10 --type charuco_board --square_size 70 --marker_size 50 -f DICT_7X7_250.json.gz --page_size B1
```

And chess board 

```bash
python gen_pattern.py -o chessboard.svg --rows 14 --columns 10 --type checkerboard --square_size 70 --page_size B1
```

And the one used to generate A4 size charuco board

```bash
python gen_pattern.py -o charuco_board.svg --rows 9 --columns 5 --type charuco_board --square_size 50 --marker_size 40 -f DICT_7X7_250.json.gz --page_size B3
```