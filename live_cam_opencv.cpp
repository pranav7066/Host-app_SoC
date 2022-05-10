#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp> 
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/types_c.h"
#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <unistd.h>
#include <linux/fb.h>
#include <linux/videodev2.h>
#include <assert.h>
#include <getopt.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include "yolo_post.h"


using namespace std;
using namespace cv;

extern image run_inference(unsigned char *in);
extern int run_xm6(void);
extern int stop_xm6(void);
extern int readfile(char *fname, unsigned char *rd, unsigned int bufsize);
extern int writefile(char *fname, unsigned char *wr, int bufsize);
extern int xm6_init(void);
extern int xm6_shutdown(void);

struct stat st;
struct fb_var_screeninfo vinfo;
struct fb_fix_screeninfo finfo;
char *fbp = 0;
int fb0 = 0;
long ssize = 0;

void display_init(){
    fb0 = open("/dev/fb0", O_RDWR);

    if (fb0 < 0) {
     printf("Error : Can not open framebuffer device/n");
     exit(1);
    }

    if (ioctl(fb0, FBIOGET_FSCREENINFO, &finfo)) {
     printf("Error reading fixed information/n");
     exit(2);
    }

    if (ioctl(fb0, FBIOGET_VSCREENINFO, &vinfo)) {
     printf("Error reading variable information/n");
     exit(3);
    }

    ssize = vinfo.xres * vinfo.yres * vinfo.bits_per_pixel / 8;
    fbp = (char*) mmap(0, ssize, PROT_READ | PROT_WRITE, MAP_SHARED, fb0, 0);
    if (fbp == (char*) -1) {
     printf("Error: failed to map framebuffer device to memory./n");
     exit(EXIT_FAILURE);
    }

    printf("Display size %d x %d %dbpp\n", vinfo.xres, vinfo.yres,
         vinfo.bits_per_pixel);
}


int main(int argc , char** argv) {
	xm6_init();
    display_init();

    char* device = "/dev/video0";
    VideoCapture camera(device);
    Mat frame, resized;

    if (!camera.isOpened()) {
        cerr << "Could not open camera" << endl;
        return 1;
    }
    camera.set(CAP_PROP_FRAME_WIDTH, 800);
    camera.set(CAP_PROP_FRAME_HEIGHT, 600);

    int frame_width = camera.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = camera.get(CAP_PROP_FRAME_HEIGHT);

    printf("Frame height: %d, width: %d\n", frame_height, frame_width);
    image im;


    for (int j = 0; j < atoi(argv[1]); j++){
        printf("Frame no: %d\n", j);
        camera >> frame;
        resize(frame, resized, Size(416, 416), INTER_LINEAR);
        unsigned char *data = (unsigned char*)(resized.data);
        // unsigned char *input = malloc(0x100000);
        // readfile("yolo_input.bin", input, 0x100000);
        im = run_inference(data);
        unsigned char *im_data = calloc(im.w * im.h * im.c, sizeof(char));

        int i, k;
        for (k = 0; k < im.c; ++k) {
            for (i = 0; i < im.w * im.h; ++i) {
                im_data[i * im.c + k] = (unsigned char) (255 * im.data[i + k * im.w * im.h]);
            }
        }

        Mat im2mat(im.w, im.h, CV_8UC3, im_data);
        Mat dis_resize;
        resize(im2mat, dis_resize, Size(1080, 1920), INTER_LINEAR);
        unsigned char *disp_data = (unsigned char*)(dis_resize.data);

        // imwrite("out_inf.jpg", dis_resize);
        printf("displaying...\n");
        if(fbp != 0){
            unsigned char *s = fbp;
            for(unsigned int i = 0; i < 1080; i++){
                for(unsigned int j = 0; j < 1920; j++){
                    *(s++)=disp_data[0];     //R
                    *(s++)=disp_data[1];     //G
                    *(s++)=disp_data[2];     //B
                    *(s++)=0xFF;        //A
                    disp_data += 3;
                }
            }
        }
        printf("display done..\n");
        free(im_data);

        free_image(im);
    }

    printf("Done!\n");
    munmap(fbp, ssize);
    close(fb0);
	// xm6_shutdown();
    return 0;
}