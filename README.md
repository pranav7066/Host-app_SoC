## Compilation

riscv64-unknown-linux-gnu/bin/riscv64-unknown-linux-gnu-g++ live_cam_opencv.cpp aiss_app.c yolo_post.c -I opencv_riscv_soc/include/opencv4/ -L opencv_riscv_soc/lib/ -lopencv_core  -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio -o live_inference -O3 -fpermissive
