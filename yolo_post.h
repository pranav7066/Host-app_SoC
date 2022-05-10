#ifndef YOLO_POST_H
#define YOLO_POST_H

typedef enum {
	PNG, BMP, TGA, JPG
} IMTYPE;
typedef struct {
	int w;
	int h;
	int c;
	float *data;
} image;
typedef struct {
	float x, y, w, h;
} box;
typedef struct node {
	void *val;
	struct node *next;
	struct node *prev;
} node;
typedef struct list {
	int size;
	node *front;
	node *back;
} list;
typedef struct detection {
	box bbox;
	int classes;
	float *prob;
	float *mask;
	float objectness;
	int sort_class;
} detection;
typedef struct layer_yolo {
	int h;
	int w;
	int f;
	float *data;
	int *mask;
} layer_yolo;
typedef struct yolo_data {
	int no_of_box;
	int classes;
	int netw;
	int neth;
	layer_yolo * yolo_0;
	layer_yolo * yolo_1;
	layer_yolo * yolo_2;
	float *biases;
} yolo_data;

void getdetections(image im, yolo_data *ydata);
void init_y(yolo_data *post_data);

void free_image(image m);
image make_empty_image(int w, int h, int c);
image make_image(int w, int h, int c);
image load_image_stb(char *filename, int channels);
image resize_image(image im, int w, int h);
image load_image(char *filename, int w, int h, int c);
image load_image_color(char *filename, int w, int h);
image copy_image(image p);
image border_image(image a, int border);
image get_image(char *buf, int w, int h, int c);

void fill_cpu(int N, float ALPHA, float *X, int INCX);
void embed_image(image source, image dest, int dx, int dy);
void composite_image(image source, image dest, int dx, int dy);
image tile_images(image a, image b, int dx);
image get_label(image **characters, char *string, int size);
void draw_box(image a, int x1, int y1, int x2, int y2, float r, float g,
		float b);
void draw_box_width(image a, int x1, int y1, int x2, int y2, int w, float r,
		float g, float b);
float get_color(int c, int x, int max);
void draw_label(image a, int r, int c, image label, const float *rgb);
image float_to_image(int w, int h, int c, float *data);
image threshold_image(image im, float thresh);
void draw_detections(image im, detection *dets, int num, float thresh,
		int classes);
void free_detections(detection *dets, int n);
void save_image_options(image im, const char *name, IMTYPE f, int quality);
void save_image(image im, const char *name);
void constrain_image(image im);
void rgbgr_image(image im);

#endif
