#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <linux/fb.h>
#include <linux/videodev2.h>
#include "yolo_post.h"
#include <sys/ioctl.h>
#include <time.h>

yolo_data ydata;

struct timeval t1, t2;

/* AISS IPC commands */
typedef enum {
	E_XM6_START_INFERENCE = 1,
	E_XM6_START_LINK_TEST = 2,
	E_XM6_START_VX_YUYV = 3,
	E_XM6_IDLE = 4,
	E_XM6_GET_STATUS = 5,
} E_IPC_CMD;

/* IPC command status */
typedef enum {
	E_XM6_CMD_OK = 0, E_XM6_CMD_FAIL = 1, E_XM6_CMD_BUSY = 2,
} E_IPC_CMD_STAT;

/* Shared memory structure */
typedef struct {
	unsigned char qdata[0x8000000];// 128Mb Q-data buffer
	unsigned char inbuff[0x400000];// 4Mb In buffer
	unsigned char outbuff[0x800000];// 8Mb Out buffer
	unsigned int cmd;// IPC Commands to process
	unsigned int cmdstat;// IPC Command status
	unsigned int im_w;// image width
	unsigned int im_h;// image height
	unsigned int outdatalen; // out buffer data length
	float mean_std_arr[8]; // input pre-process values
	int mean_std_en; // enable mean, std buffer
	float scale_factor; // input scaling factor
	int transpose_en; // input transpose flag
	float zero_point; // input zero point value
	int resize_en; // enable resize
	int resize_h; // resized height
	int resize_w;  // resized width
	int swap_en; // channel swap enable
	int profiler_en; // 0- Disabled 1- Short 2- Long 3- Dev
	float bw_reduction; // -1 Disabled,
	unsigned char linktest[16]; // buffer for link test
	unsigned int  printidx;// print index
	unsigned char printlogs[0x4000];// print logs
	unsigned char prof_buff[0x100000];// profiler buffer
}ShmData;

ShmData *shm;
unsigned char *airesmem, *pcssreg, *xm6reg;
int aiss_drv0 = -1, fd0 = -1;

int writefile(char *fname, unsigned char *wr, int bufsize) {
	FILE *fp = fopen(fname, "wb");

	if (!fp) {
		printf("file open failed %s\n", fname);
		return -1;
	}
	int fsize = fwrite(wr, bufsize, 1, fp);
	fclose(fp);
	return fsize;
}

int readfile(char *fname, unsigned char *rd, unsigned int bufsize) {
	FILE *fp = fopen(fname, "rb");

	if (!fp) {
		printf("file open failed %s\n", fname);
		return -1;
	}

	fseek(fp, 0, SEEK_END);
	int fsize = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	if (fsize < bufsize) {
		fread(rd, fsize, 1, fp);
	} else {
		printf("file size exceeds buffer size\n");
		return -1;
	}

	fclose(fp);
	return fsize;
}

void start_xm6(void) {
	memset(&shm->printlogs[0], 0, 0x4000);
	/* Start XM6 */
	*(unsigned int*) (pcssreg + 0x7CUL) = 0x1F0E01UL;
	usleep(50000u);
	*(unsigned int*) (pcssreg + 0x7CUL) = 0x1F1B01UL;
	usleep(50000u);
	*(unsigned int*) (pcssreg + 0x7CUL) = 0x1F1F01UL;
	sleep(1u);
}

void stop_xm6(void) {
	/* Put XM6 into reset state */
	*(unsigned int*) (pcssreg + 0x7CUL) = 0x1F1B01UL;
}

void load_xm6_firmware(void) {
	/* Restore XM6 firmware (copy XM6 program, data binary into DDR) */
	/* Load code section */
	readfile("xm6_prog.bin", airesmem, 0x01000000UL);
	/* Load data section */
	readfile("xm6_data.bin", airesmem + 0x01000000UL, 0x06000000UL);
}

void set_xm6_bootaddr(unsigned int addr) {
	/* Set XM6 boot vector address */
	*(unsigned int*) (pcssreg + 0x29CUL) = addr;
}

void notify_xm6(void) {
	/* Notify XM6 and wait for response. */
	*(unsigned int*) (xm6reg) = 0x01UL;
	/* Blocking read call, until there is IPC interrupt response from XM6 */
	read(aiss_drv0, NULL, 0);
}

void shm_init(void) {
	/* Shared memory start address */
	shm = (ShmData*) (airesmem + 0x07000000UL);
}

int aiss_init(void) {

	aiss_drv0 = open("/dev/aiss_drv0", O_RDWR | O_SYNC);

	if (aiss_drv0 < 0) {
		printf("open: failed /dev/aiss_drv0\n");
		return -1;
	}

	fd0 = open("/dev/mem", O_RDWR | O_SYNC);
	if (fd0 < 0) {
		printf("open: failed /dev/mem\n");
		return -1;
	}

	/* Map AISS reserved memory */
	airesmem = (unsigned char*) mmap(NULL, 0x10000000UL, PROT_READ | PROT_WRITE,
			MAP_SHARED, fd0, (off_t) 0xC50000000);

	if ((void*) airesmem == MAP_FAILED) {
		printf("fail to map airesmem memory\n");
		return -1;
	}

	/* PCSS register space */
	pcssreg = (unsigned char*) mmap(NULL, 0x10000UL, PROT_READ | PROT_WRITE,
			MAP_SHARED, fd0, (off_t) 0x4F0010000);

	if ((void*) pcssreg == MAP_FAILED) {
		printf("fail to map pcssreg memory\n");
		return -1;
	}

	/* XM6 register space */
	xm6reg = (unsigned char*) mmap(NULL, 0x10000UL, PROT_READ | PROT_WRITE,
			MAP_SHARED, fd0, (off_t) 0x262400000);

	if ((void*) xm6reg == MAP_FAILED) {
		printf("fail to map xm6reg memory\n");
		return -1;
	}

	return 0;
}

// int fb0=0;
// long ssize=0;
// volatile char *fbp = 0;

// static void open_device(void)
// {
//     struct stat st;
// 	struct fb_var_screeninfo vinfo;
// 	struct fb_fix_screeninfo finfo;
		
// 	fb0 = open ("/dev/fb0",O_RDWR);
	
// 	if (fb0 < 0) {
// 	printf("Error : Can not open framebuffer device/n");
// 	exit(1);
// 	}
	
// 	if (ioctl(fb0,FBIOGET_FSCREENINFO,&finfo)) {
// 	printf("Error reading fixed information/n");
// 	exit(2);
// 	}
// 	if (ioctl(fb0,FBIOGET_VSCREENINFO,&vinfo)) {
// 	printf("Error reading variable information/n");
// 	exit(3);
// 	}
// 	ssize = vinfo.xres * vinfo.yres * vinfo.bits_per_pixel / 8;
// 	fbp =(char *) mmap (0, ssize, PROT_READ | PROT_WRITE, MAP_SHARED, fb0,0);
// 	if (fbp == (char *)-1)
// 	{
// 		printf ("Error: failed to map framebuffer device to memory./n");
// 		exit (EXIT_FAILURE);
// 	}
	
// 	printf("ssize %d x %d %dbpp\n",vinfo.xres, vinfo.yres, vinfo.bits_per_pixel);
// }

unsigned char* do_vx(unsigned char *yuv_in, int w, int h)
{	
	start_xm6();
	printf("start VX\n");
	/* read image */
	memcpy(&shm->inbuff[0], yuv_in, w*h*2);
	shm->im_h = h;
	shm->im_w = w;
	shm->cmd = E_XM6_START_VX_YUYV;

	notify_xm6();
	printf("done\n");
	stop_xm6();
	return shm->outbuff;
}

image run_inference(unsigned char *in)
{
	start_xm6();
	char out[20];
	static int count = 0;
	double elapsed_time;

	/* read image */
	memcpy(&shm->inbuff[0], in, 416*416*3);
	shm->im_h = 416;
	shm->im_w = 416;

	shm->cmd = E_XM6_START_INFERENCE;

	// printf("%s", &shm->printlogs[0]);
	shm->printidx = 0;

	gettimeofday(&t1, NULL);
	notify_xm6();
	gettimeofday(&t2, NULL);
	// printf("%s", &shm->printlogs[0]);
	printf("Inference done !\n");

	elapsed_time = (t2.tv_sec - t1.tv_sec) * 1000;
	elapsed_time += (t2.tv_usec - t1.tv_usec) / 1000;
	printf("Elapsed time for inference: %f ms.\n", elapsed_time);

	image im = get_image(&shm->inbuff[0], shm->im_w, shm->im_h, 3);

	gettimeofday(&t1, NULL);
	getdetections(im, &ydata);
	gettimeofday(&t2, NULL);

	elapsed_time = (t2.tv_sec - t1.tv_sec) * 1000;
	elapsed_time += (t2.tv_usec - t1.tv_usec) / 1000;
	printf("Elapsed time for post-processing: %f ms.\n", elapsed_time);

	// printf("Saving predictions ..\n");	
	// image resized = resize_image(im, 1080, 1920);
	// count++;
	// sprintf(out, "yolov3_out_%d.jpg", count);
	// save_image(resized, out);
	// free_image(resized);
	stop_xm6();
	return im;

	// free(in);
}

int xm6_init(void)
{
	printf("AISS camera capture !!\n");

	/* Initialize AISS related memory mappings */
	if (aiss_init() != 0) {
		return -1;
	}

	/* Set XM6 boot vector address and load the firmware. */
	stop_xm6();
	set_xm6_bootaddr(0x50000000UL);
	load_xm6_firmware();
	printf("XM6 firmware loaded !!\n");

	/* Initialize shared memory */
	shm_init();

	/* Clear profiler buffer */
	memset(&shm->prof_buff[0], 0, 0x100000);

	double elapsed_time;

	printf("YOLOv3 run\n");
	/* Initialize yolo post process */
	init_y(&ydata);

	ydata.yolo_0->data = (float*) (shm->outbuff); //bin0
	ydata.yolo_1->data = (float*) (shm->outbuff + (13 * 13 * 255 * 4)); //bin1
	ydata.yolo_2->data = (float*) (shm->outbuff + (26 * 26 * 255 * 4)); //bin2

	/* Read image */
	//readfile("yolo_input.bin", &shm->inbuff[0], 0x400000UL);
	/* Parameters must match deploy CSV values, used during qdata generation */
	shm->im_h = 416;
	shm->im_w = 416;
	shm->swap_en = 0;
	shm->resize_en = 0;
	shm->resize_h = 0;
	shm->resize_w = 0;
	shm->transpose_en = 0;
	shm->zero_point = 0;
	shm->bw_reduction = -1;
	shm->profiler_en = 1;
	shm->mean_std_en = 0;
	shm->scale_factor = (float)1 / 256; // 1/256 for yolo
	
	/* Read qdata */
	printf("Loading YOLOv3...\n");
	readfile("yolov3.deploy.cdnnQdata", &shm->qdata[0], 0x8000000UL);
	printf("Yolov3 loaded...\n");
	start_xm6();
	return 0;
}

int xm6_shutdown(void)
{
	stop_xm6();
	// close(fd0);
	munmap(airesmem, 0x10000000UL);
	munmap(pcssreg, 0x10000UL);
	munmap(xm6reg, 0x10000UL);
	close(aiss_drv0);

}

#if 0
int main(char argc, char *argv[]) {
	char *opt = argv[1];
	opt++;
	// open_device();
	printf("AISS host app start.\n");

	/* Initialize AISS related memory mappings */
	if (aiss_init() != 0) {
		return -1;
	}

	/* Set XM6 boot vector address and load the firmware. */
	stop_xm6();
	set_xm6_bootaddr(0x50000000UL);
	load_xm6_firmware();
	printf("XM6 firmware loaded...\n");


	/* Initialize shared memory */
	shm_init();

	/* Clear profiler buffer */
	memset(&shm->prof_buff[0], 0, 0x100000);

	double elapsed_time;

	switch (*opt) {
	case 'c':

		printf("Image Classification run\n");
		
		if (strcmp(argv[2], "-mv1") == 0){
			printf("MobilenetV1 selected\n");
			/* Read image */
			readfile("mb_input.bin", &shm->inbuff[0], 0x400000UL);
			/* Parameters must match deploy CSV values, used during qdata generation */

			shm->im_h = 224;
			shm->im_w = 224;
			shm->swap_en = 0;
			shm->resize_en = 0;
			shm->resize_h = 0;
			shm->resize_w = 0;
			shm->transpose_en = 0;
			shm->zero_point = 0;
			shm->bw_reduction = -1;
			shm->profiler_en = 1;
			float f32Arr[8] = { 127, 127, 127, -1, 127, 127, 127, -1 };//mobile-net
			memcpy(shm->mean_std_arr, f32Arr, sizeof(f32Arr));
			shm->mean_std_en = 1;
			shm->scale_factor = 1; // 1 for mobile-net
		
			/* Read qdata */
			readfile("mobilenetv1.deploy.cdnnQdata", &shm->qdata[0], 0x8000000UL);
			printf("MobilenetV1 loaded...\n");

			printf("Start inference ..\n");
			shm->cmd = E_XM6_START_INFERENCE;
			start_xm6();
			printf("%s", &shm->printlogs[0]);
			shm->printidx = 0;
			gettimeofday(&t1, NULL);
			notify_xm6();
			gettimeofday(&t2, NULL);
			printf("%s", &shm->printlogs[0]);
			
			elapsed_time = (t2.tv_sec - t1.tv_sec) * 1000;
			elapsed_time += (t2.tv_usec - t1.tv_usec) / 1000;
			printf("Elapsed time for inference: %f ms.\n", elapsed_time);

			
			overlay(&shm->inbuff[0], shm->im_w, shm->im_h, &shm->printlogs[54], "mobilenetv1_out.jpg");


			/* Write profiler data */
			writefile("mobilenetv1_profiler.xls", &shm->prof_buff[0],
					strlen(&shm->prof_buff[0]));
		}

		else if (strcmp(argv[2], "-mv2") == 0){
			printf("MobileNetV2 selected\n");
			/* Read image */
			readfile("mb_input.bin", &shm->inbuff[0], 0x400000UL);
			/* Parameters must match deploy CSV values, used during qdata generation */
			shm->im_h = 224;
			shm->im_w = 224;
			shm->swap_en = 0;
			shm->resize_en = 0;
			shm->resize_h = 0;
			shm->resize_w = 0;
			shm->transpose_en = 0;
			shm->zero_point = 0;
			shm->bw_reduction = -1;
			shm->profiler_en = 1;
			float f32Arr[8] = { 127, 127, 127, -1, 127, 127, 127, -1 };
			memcpy(shm->mean_std_arr, f32Arr, sizeof(f32Arr));
			shm->mean_std_en = 1;
			shm->scale_factor = 1; // 1 for mobile-net
		
			/* Read qdata */
			readfile("mobilenetv2.deploy.cdnnQdata", &shm->qdata[0], 0x8000000UL);
			printf("MobileNetV2 loaded...\n");

			printf("Start inference ..\n");
			shm->cmd = E_XM6_START_INFERENCE;
			start_xm6();
			printf("%s", &shm->printlogs[0]);
			shm->printidx = 0;
			gettimeofday(&t1, NULL);
			notify_xm6();
			gettimeofday(&t2, NULL);
			printf("%s", &shm->printlogs[0]);

			elapsed_time = (t2.tv_sec - t1.tv_sec) * 1000;
			elapsed_time += (t2.tv_usec - t1.tv_usec) / 1000;
			printf("Elapsed time for inference: %f ms.\n", elapsed_time);
			overlay(&shm->inbuff[0], shm->im_w, shm->im_h, &shm->printlogs[54], "mobilenetv2_out.jpg");

			/* Write profiler data */
			writefile("mobilenetv2_profiler.xls", &shm->prof_buff[0],
					strlen(&shm->prof_buff[0]));
		}

		else if (strcmp(argv[2], "-iv4") == 0){
			printf("InceptionV4 selected\n");
			/* Read image */
			readfile("inceptionv4_input.bin", &shm->inbuff[0], 0x400000UL);
			/* Parameters must match deploy CSV values, used during qdata generation */
			shm->im_h = 299;
			shm->im_w = 299;
			shm->swap_en = 0;
			shm->resize_en = 0;
			shm->resize_h = 0;
			shm->resize_w = 0;
			shm->transpose_en = 0;
			shm->zero_point = 0;
			shm->bw_reduction = -1;
			shm->profiler_en = 1;
			shm->mean_std_en = 0;
			shm->scale_factor = (float)1 / 256; // 1 for mobile-net
		
			/* Read qdata */
			readfile("inceptionv4.deploy.cdnnQdata", &shm->qdata[0], 0x8000000UL);
			printf("InceptionV4 loaded...\n");

			printf("Start inference ..\n");
			shm->cmd = E_XM6_START_INFERENCE;
			start_xm6();
			printf("%s", &shm->printlogs[0]);
			shm->printidx = 0;
			gettimeofday(&t1, NULL);
			notify_xm6();
			gettimeofday(&t2, NULL);
			printf("%s", &shm->printlogs[0]);

			elapsed_time = (t2.tv_sec - t1.tv_sec) * 1000;
			elapsed_time += (t2.tv_usec - t1.tv_usec) / 1000;
			printf("Elapsed time for inference: %f ms.\n", elapsed_time);

			overlay(&shm->inbuff[0], shm->im_w, shm->im_h, &shm->printlogs[54], "inceptionv4_out.jpg");


			/* Write profiler data */
			writefile("inceptionv4_profiler.xls", &shm->prof_buff[0],
					strlen(&shm->prof_buff[0]));
		}


		else if (strcmp(argv[2], "-enl") == 0){
			printf("EfficientNet_Edge_TPU-L selected\n");
			/* Read image */
			readfile("efficientnet-l_input.bin", &shm->inbuff[0], 0x400000UL);
			/* Parameters must match deploy CSV values, used during qdata generation */
			shm->im_h = 300;
			shm->im_w = 300;
			shm->swap_en = 0;
			shm->resize_en = 0;
			shm->resize_h = 0;
			shm->resize_w = 0;
			shm->transpose_en = 0;
			shm->zero_point = 0;
			shm->bw_reduction = -1;
			shm->profiler_en = 1;
			shm->mean_std_en = 0;
			shm->scale_factor = (float)1 / 256; // 1 for mobile-net
		
			/* Read qdata */
			readfile("efficientnet-L.deploy.cdnnQdata", &shm->qdata[0], 0x8000000UL);
			printf("EfficientNet_Edge_TPU-L loaded...\n");

			printf("Start inference ..\n");
			shm->cmd = E_XM6_START_INFERENCE;
			start_xm6();
			printf("%s", &shm->printlogs[0]);
			shm->printidx = 0;
			gettimeofday(&t1, NULL);
			notify_xm6();
			gettimeofday(&t2, NULL);
			printf("%s", &shm->printlogs[0]);
			
			elapsed_time = (t2.tv_sec - t1.tv_sec) * 1000;
			elapsed_time += (t2.tv_usec - t1.tv_usec) / 1000;
			printf("Elapsed time for inference: %f ms.\n", elapsed_time);

			/* Write profiler data */
			writefile("efficientnet-l_profiler.xls", &shm->prof_buff[0],
					strlen(&shm->prof_buff[0]));
		}

		else{
			printf("No model selected...\n");
		}

		break;

	case 'd':

		printf("YOLOv3 run\n");
		/* Initialize yolo post process */
		init_y(&ydata);

		ydata.yolo_0->data = (float*) (shm->outbuff); //bin0
		ydata.yolo_1->data = (float*) (shm->outbuff + (13 * 13 * 255 * 4)); //bin1
		ydata.yolo_2->data = (float*) (shm->outbuff + (26 * 26 * 255 * 4)); //bin2

		/* Read image */
		readfile("yolo_input.bin", &shm->inbuff[0], 0x400000UL);
		/* Parameters must match deploy CSV values, used during qdata generation */
		shm->im_h = 416;
		shm->im_w = 416;
		shm->swap_en = 0;
		shm->resize_en = 0;
		shm->resize_h = 0;
		shm->resize_w = 0;
		shm->transpose_en = 0;
		shm->zero_point = 0;
		shm->bw_reduction = -1;
		shm->profiler_en = 1;
		shm->mean_std_en = 0;
		shm->scale_factor = (float)1 / 256; // 1/256 for yolo
		
		/* Read qdata */
		readfile("yolov3.deploy.cdnnQdata", &shm->qdata[0], 0x8000000UL);
		printf("Yolov3 loaded...\n");
		start_xm6();








		/* Write profiler data */
		//writefile("yolov3_profiler.xls", &shm->prof_buff[0],
		//strlen(&shm->prof_buff[0]));
		break;

	case 'l':
		printf("Link test \n");
		start_xm6();
		printf("%s", &shm->printlogs[0]);
		for (int i = 0; i < 5; i++) {
			shm->printidx = 0;
			shm->cmd = E_XM6_START_LINK_TEST;
			sprintf(shm->linktest, "message %d", i);
			printf("\nU74->XM6 message %d\n", i);

			notify_xm6();
			printf("U74<-XM6 %s\n", &shm->linktest[0]);
		}
		break;

	case 'v':
		printf("VX test \n");

		/* read image */
		readfile("./vx_input.raw", &shm->inbuff[0], 0x400000UL);
		shm->im_h = 176;
		shm->im_w = 144;
		shm->cmd = E_XM6_START_VX_YUYV;

		start_xm6();
		notify_xm6();

		writefile("./vx_out.raw", shm->outbuff, shm->im_w * shm->im_h * 3);
		printf("done !\n");
		break;

	case 'p':
		printf("%s", &shm->printlogs[0]);
		break;
	default:
		break;
	}

	stop_xm6();
	close(aiss_drv0);
	close(fd0);
	// munmap (fbp, ssize);
 //    close(fb0);
	munmap(airesmem, 0x10000000UL);
	munmap(pcssreg, 0x10000UL);
	munmap(xm6reg, 0x10000UL);
	return 0;
}
#endif
