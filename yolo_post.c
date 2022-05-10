#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "yolo_post.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

extern int writefile(char *fname, unsigned char *wr, int bufsize);

#define exp expf_fast

float colors[6][3] = { { 1, 0, 1 }, { 0, 0, 1 }, { 0, 1, 1 }, { 0, 1, 0 }, { 1,
		1, 0 }, { 1, 0, 0 } };
static image label[80];
static char *names[] = { "person", "bicycle", "car", "motorbike", "aeroplane",
		"bus", "train", "truck", "boat", "traffic light", "fire hydrant",
		"stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
		"sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
		"umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
		"snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
		"skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
		"cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
		"orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
		"chair", "sofa", "pottedplant", "bed", "diningtable", "toilet",
		"tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
		"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
		"vase", "scissors", "teddy bear", "hair drier", "toothbrush" };
static float get_pixel(image m, int x, int y, int c);
static void set_pixel(image m, int x, int y, int c, float val);
static void add_pixel(image m, int x, int y, int c, float val);
static float get_pixel_extend(image m, int x, int y, int c);
static void load_labels(void);

static int entry_index(int w, int h, int location, int classes, int entry);
static float logistic_activate(float x);
static void activate_array(float *x, const int n);
static void forward_yolo_layer(int h, int w, int classes, float * input,
		int no_of_box);
static void forward_yolo(yolo_data * post_data);
static int yolo_num_detections(layer_yolo *yolo_n, float thresh, int no_of_box,
		int classes);
static int num_detections(yolo_data * post_yolo, float thresh);
static detection *make_network_boxes(yolo_data * post_yolo, float thresh,
		int *num);
static box get_yolo_box(float *x, float *biases, int n, int index, int i, int j,
		int lw, int lh, int w, int h, int stride);
static void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw,
		int neth, int relative);
static int get_yolo_detections(layer_yolo * yolo_n, int w, int h, int netw,
		int neth, float thresh, int *map, int relative, detection *dets,
		int no_of_box, int classes, float *biases);
static void fill_network_boxes(yolo_data * post_yolo, int w, int h,
		float thresh, float hier, int *map, int relative, detection *dets);
static detection *get_network_boxes(yolo_data * post_yolo, int w, int h,
		float thresh, float hier, int *map, int relative, int *num);
static int nms_comparator(const void *pa, const void *pb);
static float overlap(float x1, float w1, float x2, float w2);
static float box_intersection(box a, box b);
static float box_union(box a, box b);
static float box_iou(box a, box b);
static void do_nms_sort(detection *dets, int total, int classes, float thresh);

/*-----LoadImage + LOAD names+ load alphabet-----*/
void free_image(image m) {
	if (m.data) {
		free(m.data);
	}
}

static inline float expf_fast(float a) {
  	union { float f; int x; } u;
  	u.x = (int) (12102203 * a + 1064866805);
  	return u.f;
}

static float get_pixel(image m, int x, int y, int c) {
	assert(x < m.w && y < m.h && c < m.c);
	return m.data[c * m.h * m.w + y * m.w + x];
}

static void set_pixel(image m, int x, int y, int c, float val) {
	if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c)
		return;
	assert(x < m.w && y < m.h && c < m.c);
	m.data[c * m.h * m.w + y * m.w + x] = val;
}

static void add_pixel(image m, int x, int y, int c, float val) {
	assert(x < m.w && y < m.h && c < m.c);
	m.data[c * m.h * m.w + y * m.w + x] += val;
}

image make_empty_image(int w, int h, int c) {
	image out;
	out.data = 0;
	out.h = h;
	out.w = w;
	out.c = c;
	return out;
}

image make_image(int w, int h, int c) {
	image out = make_empty_image(w, h, c);
	out.data = calloc(h * w * c, sizeof(float));
	return out;
}

image load_image_stb(char *filename, int channels) {
	int w, h, c;
	unsigned char *data = stbi_load(filename, &w, &h, &c, channels);
	if (!data) {
		fprintf(stderr, "Cannot load image \"%s\"\nSTB Reason: %s\n", filename,
				stbi_failure_reason());
		exit(0);
	}
	if (channels)
		c = channels;
	int i, j, k;
	image im = make_image(w, h, c);
	for (k = 0; k < c; ++k) {
		for (j = 0; j < h; ++j) {
			for (i = 0; i < w; ++i) {
				int dst_index = i + w * j + w * h * k;
				int src_index = k + c * i + c * w * j;
				im.data[dst_index] = (float) data[src_index] / 255.;
			}
		}
	}
	free(data);
	return im;
}

image resize_image(image im, int w, int h) {
	image resized = make_image(w, h, im.c);
	image part = make_image(w, im.h, im.c);
	int r, c, k;
	float w_scale = (float) (im.w - 1) / (w - 1);
	float h_scale = (float) (im.h - 1) / (h - 1);
	for (k = 0; k < im.c; ++k) {
		for (r = 0; r < im.h; ++r) {
			for (c = 0; c < w; ++c) {
				float val = 0;
				if (c == w - 1 || im.w == 1) {
					val = get_pixel(im, im.w - 1, r, k);
				} else {
					float sx = c * w_scale;
					int ix = (int) sx;
					float dx = sx - ix;
					val = (1 - dx) * get_pixel(im, ix, r, k)
							+ dx * get_pixel(im, ix + 1, r, k);
				}
				set_pixel(part, c, r, k, val);
			}
		}
	}
	for (k = 0; k < im.c; ++k) {
		for (r = 0; r < h; ++r) {
			float sy = r * h_scale;
			int iy = (int) sy;
			float dy = sy - iy;
			for (c = 0; c < w; ++c) {
				float val = (1 - dy) * get_pixel(part, c, iy, k);
				set_pixel(resized, c, r, k, val);
			}
			if (r == h - 1 || im.h == 1)
				continue;
			for (c = 0; c < w; ++c) {
				float val = dy * get_pixel(part, c, iy + 1, k);
				add_pixel(resized, c, r, k, val);
			}
		}
	}

	free_image(part);
	return resized;
}

image load_image(char *filename, int w, int h, int c) {
	image out = load_image_stb(filename, c);
	if ((h && w) && (h != out.h || w != out.w)) {
		image resized = resize_image(out, w, h);
		free_image(out);
		out = resized;
	}
	return out;
}

image get_image(char *buf, int w, int h, int c) {
	int i, j, k;
	image im = make_image(w, h, c);
	
	for (k = 0; k < c; ++k) {
		for (j = 0; j < h; ++j) {
			for (i = 0; i < w; ++i) {
				int dst_index = i + w * j + w * h * k;
				int src_index = k + c * i + c * w * j;
				im.data[dst_index] = (float) buf[src_index] / 255.;
			}
		}
	}
	return im;
}

image load_image_color(char *filename, int w, int h) {
	return load_image(filename, w, h, 3);
}

/*----draw detections-----*/
static float get_pixel_extend(image m, int x, int y, int c) {
	if (x < 0 || x >= m.w || y < 0 || y >= m.h)
		return 0;
	/*
	 if(x < 0) x = 0;
	 if(x >= m.w) x = m.w-1;
	 if(y < 0) y = 0;
	 if(y >= m.h) y = m.h-1;
	 */
	if (c < 0 || c >= m.c)
		return 0;
	return get_pixel(m, x, y, c);
}

image copy_image(image p) {
	image copy = p;
	copy.data = calloc(p.h * p.w * p.c, sizeof(float));
	memcpy(copy.data, p.data, p.h * p.w * p.c * sizeof(float));
	return copy;
}

void constrain_image(image im)
{
    int i;
    for(i = 0; i < im.w*im.h*im.c; ++i){
        if(im.data[i] < 0) im.data[i] = 0;
        if(im.data[i] > 1) im.data[i] = 1;
    }
}

void rgbgr_image(image im)
{
    int i;
    for(i = 0; i < im.w*im.h; ++i){
        float swap = im.data[i];
        im.data[i] = im.data[i+im.w*im.h*2];
        im.data[i+im.w*im.h*2] = swap;
    }
}

image border_image(image a, int border) {
	image b = make_image(a.w + 2 * border, a.h + 2 * border, a.c);
	int x, y, k;
	for (k = 0; k < b.c; ++k) {
		for (y = 0; y < b.h; ++y) {
			for (x = 0; x < b.w; ++x) {
				float val = get_pixel_extend(a, x - border, y - border, k);
				if (x - border < 0 || x - border >= a.w || y - border < 0
						|| y - border >= a.h)
					val = 1;
				set_pixel(b, x, y, k, val);
			}
		}
	}
	return b;
}

void fill_cpu(int N, float ALPHA, float *X, int INCX) {
	int i;
	for (i = 0; i < N; ++i)
		X[i * INCX] = ALPHA;
}

void embed_image(image source, image dest, int dx, int dy) {
	int x, y, k;
	for (k = 0; k < source.c; ++k) {
		for (y = 0; y < source.h; ++y) {
			for (x = 0; x < source.w; ++x) {
				float val = get_pixel(source, x, y, k);
				set_pixel(dest, dx + x, dy + y, k, val);
			}
		}
	}
}

void composite_image(image source, image dest, int dx, int dy) {
	int x, y, k;
	for (k = 0; k < source.c; ++k) {
		for (y = 0; y < source.h; ++y) {
			for (x = 0; x < source.w; ++x) {
				float val = get_pixel(source, x, y, k);
				float val2 = get_pixel_extend(dest, dx + x, dy + y, k);
				set_pixel(dest, dx + x, dy + y, k, val * val2);
			}
		}
	}
}

image tile_images(image a, image b, int dx) {
	if (a.w == 0)
		return copy_image(b);
	image c = make_image(a.w + b.w + dx, (a.h > b.h) ? a.h : b.h,
			(a.c > b.c) ? a.c : b.c);
	fill_cpu(c.w * c.h * c.c, 1, c.data, 1);
	embed_image(a, c, 0, 0);
	composite_image(b, c, a.w + dx, 0);
	return c;
}

static void load_labels(void) {
	char fname[25];

	for (int i = 0; i < 80; i++) {
		sprintf(fname, "./labels/%s.png", names[i]);
		label[i] = load_image(fname, 0, 0, 3);
	}
}

void draw_box(image a, int x1, int y1, int x2, int y2, float r, float g,
		float b) {
	//normalize_image(a);
	int i;
	if (x1 < 0)
		x1 = 0;
	if (x1 >= a.w)
		x1 = a.w - 1;
	if (x2 < 0)
		x2 = 0;
	if (x2 >= a.w)
		x2 = a.w - 1;

	if (y1 < 0)
		y1 = 0;
	if (y1 >= a.h)
		y1 = a.h - 1;
	if (y2 < 0)
		y2 = 0;
	if (y2 >= a.h)
		y2 = a.h - 1;

	for (i = x1; i <= x2; ++i) {
		a.data[i + y1 * a.w + 0 * a.w * a.h] = r;
		a.data[i + y2 * a.w + 0 * a.w * a.h] = r;

		a.data[i + y1 * a.w + 1 * a.w * a.h] = g;
		a.data[i + y2 * a.w + 1 * a.w * a.h] = g;

		a.data[i + y1 * a.w + 2 * a.w * a.h] = b;
		a.data[i + y2 * a.w + 2 * a.w * a.h] = b;
	}
	for (i = y1; i <= y2; ++i) {
		a.data[x1 + i * a.w + 0 * a.w * a.h] = r;
		a.data[x2 + i * a.w + 0 * a.w * a.h] = r;

		a.data[x1 + i * a.w + 1 * a.w * a.h] = g;
		a.data[x2 + i * a.w + 1 * a.w * a.h] = g;

		a.data[x1 + i * a.w + 2 * a.w * a.h] = b;
		a.data[x2 + i * a.w + 2 * a.w * a.h] = b;
	}
}

void draw_box_width(image a, int x1, int y1, int x2, int y2, int w, float r,
		float g, float b) {
	int i;
	for (i = 0; i < w; ++i) {
		draw_box(a, x1 + i, y1 + i, x2 - i, y2 - i, r, g, b);
	}
}

float get_color(int c, int x, int max) {
	float ratio = ((float) x / max) * 5;
	int i = floor(ratio);
	int j = ceil(ratio);
	ratio -= i;
	float r = (1 - ratio) * colors[i][c] + ratio * colors[j][c];
	//printf("%f\n", r);
	return r;
}

void draw_label(image a, int r, int c, image label, const float *rgb) {
	int w = label.w;
	int h = label.h;
	if (r - h >= 0)
		r = r - h;

	int i, j, k;
	for (j = 0; j < h && j + r < a.h; ++j) {
		for (i = 0; i < w && i + c < a.w; ++i) {
			for (k = 0; k < label.c; ++k) {
				float val = get_pixel(label, i, j, k);
				set_pixel(a, i + c, j + r, k, rgb[k] * val);
			}
		}
	}
}

image float_to_image(int w, int h, int c, float *data) {
	image out = make_empty_image(w, h, c);
	out.data = data;
	return out;
}

image threshold_image(image im, float thresh) {
	int i;
	image t = make_image(im.w, im.h, im.c);
	for (i = 0; i < im.w * im.h * im.c; ++i) {
		t.data[i] = im.data[i] > thresh ? 1 : 0;
	}
	return t;
}

void draw_detections(image im, detection *dets, int num, float thresh,
		int classes) {
	for (int i = 0; i < num; ++i) {
		for (int j = 0; j < classes; ++j) {
			if (dets[i].prob[j] > thresh) {
				printf("%s: %.0f%%\n", names[j], dets[i].prob[j] * 100);

				int width = im.h * .007;
				int offset = j * 123457 % classes;
				float red = get_color(2, offset, classes);
				float green = get_color(1, offset, classes);
				float blue = get_color(0, offset, classes);
				float rgb[3];

				rgb[0] = red;
				rgb[1] = green;
				rgb[2] = blue;
				box b = dets[i].bbox;
				int left = (b.x - b.w / 2.) * im.w;
				int right = (b.x + b.w / 2.) * im.w;
				int top = (b.y - b.h / 2.) * im.h;
				int bot = (b.y + b.h / 2.) * im.h;

				if (left < 0)
					left = 0;
				if (right > im.w - 1)
					right = im.w - 1;
				if (top < 0)
					top = 0;
				if (bot > im.h - 1)
					bot = im.h - 1;
				draw_box_width(im, left, top, right, bot, width, red, green,
						blue);
				draw_label(im, top, left, label[j], rgb);
				break;
			}
		}
	}
}

void free_detections(detection *dets, int n) {
	int i;
	for (i = 0; i < n; ++i) {
		free(dets[i].prob);
		if (dets[i].mask)
			free(dets[i].mask);
	}
	free(dets);
}

int rgb888torgb565(unsigned char *rgb888_buf, int rgb888_size,
		unsigned short *rgb565_buf, int rgb565_size) {
	unsigned char Red = 0, Green = 0, Blue = 0;
	int count = 0;

	if (rgb888_buf == NULL || rgb888_size <= 0 || rgb565_buf == NULL
			|| rgb565_size <= 0 || (rgb565_size < (rgb888_size / 3) * 2)) {
		printf("Invalid input parameter in %s\n", __FUNCTION__);
		return -1;
	}

	for (int i = 0; i < rgb888_size; i += 3) {
		Red = rgb888_buf[i] >> 3;
		Green = rgb888_buf[i + 1] >> 2;
		Blue = rgb888_buf[i + 2] >> 3;
		rgb565_buf[count++] = ((Red << 11) | (Green << 5) | (Blue));
	}
	return count;
}

// extern volatile char *fbp;

void save_image_options(image im, const char *name, IMTYPE f, int quality) {
	char buff[256];

	if (f == PNG)
		sprintf(buff, "%s.png", name);
	else if (f == BMP)
		sprintf(buff, "%s.bmp", name);
	else if (f == TGA)
		sprintf(buff, "%s.tga", name);
	else if (f == JPG)
		sprintf(buff, "%s.jpg", name);
	else
		sprintf(buff, "%s.png", name);

	unsigned char *data = calloc(im.w * im.h * im.c, sizeof(char));
	int i, k;
	for (k = 0; k < im.c; ++k) {
		for (i = 0; i < im.w * im.h; ++i) {
			data[i * im.c + k] = (unsigned char) (255 * im.data[i + k * im.w * im.h]);
		}
	}
	// writefile("raw.bin", data, im.w * im.h * im.c);

	int success = 0;
	if (f == PNG)
		success = stbi_write_png(buff, im.w, im.h, im.c, data, im.w * im.c);
	else if (f == BMP)
		success = stbi_write_bmp(buff, im.w, im.h, im.c, data);
	else if (f == TGA)
		success = stbi_write_tga(buff, im.w, im.h, im.c, data);
	else if (f == JPG)
		success = stbi_write_jpg(buff, im.w, im.h, im.c, data, quality);

	 // int size_565 = im.w * im.h * 2;
	 // unsigned short* buff_565 = (unsigned short*)malloc(size_565);
	 // rgb888torgb565(data, im.w*im.h*im.c, buff_565, size_565);

	 // fwrite(buff_565, size_565, 1, fp);
	 // fclose(fp);
	 
	 /* send image to display frame buffer */

	// free(data);
	// free(a);
}

void save_image(image im, const char *name) {
	save_image_options(im, name, JPG, 80);
}

/*------ functions for applying yolo layer for------- */
static int entry_index(int w, int h, int location, int classes, int entry) {
	int n = location / (w * h);
	int loc = location % (w * h);
	return n * w * h * (4 + classes + 1) + entry * w * h + loc;
}
static float logistic_activate(float x) {
	return 1. / (1. + exp(-x));
}
static void activate_array(float *x, const int n) {
	int i;
	for (i = 0; i < n; ++i) {
		x[i] = logistic_activate(x[i]);
	}
}

static void forward_yolo_layer(int h, int w, int classes, float * input,
		int no_of_box) {
	for (int n = 0; n < no_of_box; ++n) {
		int index = entry_index(w, h, n * w * h, classes, 0);
		activate_array(input + index, 2 * w * h);
		index = entry_index(w, h, n * w * h, classes, 4);
		activate_array(input + index, (1 + classes) * w * h);
	}
}

static void forward_yolo(yolo_data * post_data) {
	forward_yolo_layer(post_data->yolo_0->h, post_data->yolo_0->w,
			post_data->classes, post_data->yolo_0->data, post_data->no_of_box);
	forward_yolo_layer(post_data->yolo_1->h, post_data->yolo_1->w,
			post_data->classes, post_data->yolo_1->data, post_data->no_of_box);
	forward_yolo_layer(post_data->yolo_2->h, post_data->yolo_2->w,
			post_data->classes, post_data->yolo_2->data, post_data->no_of_box);
}
/*----- functions to get network boxes------ */

static int yolo_num_detections(layer_yolo *yolo_n, float thresh, int no_of_box,
		int classes) {
	int i, n;
	int count = 0;
	for (i = 0; i < yolo_n->w * yolo_n->h; ++i) {
		for (n = 0; n < no_of_box; ++n) {
			int obj_index = entry_index(yolo_n->w, yolo_n->h,
					n * yolo_n->w * yolo_n->h + i, classes, 4);
			if (yolo_n->data[obj_index] > thresh) {
				++count;
			}
		}
	}
	return count;
}

static int num_detections(yolo_data * post_yolo, float thresh) {
	int s = 0;
	s += yolo_num_detections(post_yolo->yolo_0, thresh, post_yolo->no_of_box,
			post_yolo->classes);
	s += yolo_num_detections(post_yolo->yolo_1, thresh, post_yolo->no_of_box,
			post_yolo->classes);
	s += yolo_num_detections(post_yolo->yolo_2, thresh, post_yolo->no_of_box,
			post_yolo->classes);
	return s;
}

static detection *make_network_boxes(yolo_data * post_yolo, float thresh,
		int *num) {
	/* make_network_boxes is going to find no of boxes.
	 * by checking all bounding boxes confidence score which are more than threshold*/
	/*once it finds N no of bounding boxes , then it creates N no of  empty bounding boxes */
	int nboxes = num_detections(post_yolo, thresh);
	if (num)
		*num = nboxes;
	detection *dets = calloc(nboxes, sizeof(detection));
	for (int i = 0; i < nboxes; ++i) {
		dets[i].prob = calloc(post_yolo->classes, sizeof(float));
	}
	return dets;
}
static box get_yolo_box(float *x, float *biases, int n, int index, int i, int j,
		int lw, int lh, int w, int h, int stride) {
	box b;
	b.x = (i + x[index + 0 * stride]) / lw;
	b.y = (j + x[index + 1 * stride]) / lh;
	b.w = exp(x[index + 2 * stride]) * biases[2 * n] / w;
	b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
	return b;
}
static void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw,
		int neth, int relative) {
	int i;
	int new_w = 0;
	int new_h = 0;
	if (((float) netw / w) < ((float) neth / h)) {
		new_w = netw;
		new_h = (h * netw) / w;
	} else {
		new_h = neth;
		new_w = (w * neth) / h;
	}
	for (i = 0; i < n; ++i) {
		box b = dets[i].bbox;
		b.x = (b.x - (netw - new_w) / 2. / netw) / ((float) new_w / netw);
		b.y = (b.y - (neth - new_h) / 2. / neth) / ((float) new_h / neth);
		b.w *= (float) netw / new_w;
		b.h *= (float) neth / new_h;
		if (!relative) {
			b.x *= w;
			b.w *= w;
			b.y *= h;
			b.h *= h;
		}
		dets[i].bbox = b;
	}
}

static int get_yolo_detections(layer_yolo * yolo_n, int w, int h, int netw,
		int neth, float thresh, int *map, int relative, detection *dets,
		int no_of_box, int classes, float *biases) {
	/*This function calls get_yolo_box function,
	 * get_yolo_box gets bounding box values x,y coordinates
	 * and w and H of bounding box */
	int i, j, n;
	float *predictions = yolo_n->data;
	int count = 0;
	for (i = 0; i < yolo_n->w * yolo_n->h; ++i) {
		int row = i / yolo_n->w;
		int col = i % yolo_n->w;
		for (n = 0; n < no_of_box; ++n) {
			int obj_index = entry_index(yolo_n->w, yolo_n->h,
					n * yolo_n->w * yolo_n->h + i, classes, 4);
			float objectness = predictions[obj_index];
			if (objectness <= thresh)
				continue;
			int box_index = entry_index(yolo_n->w, yolo_n->h,
					n * yolo_n->w * yolo_n->h + i, classes, 0);
			dets[count].bbox = get_yolo_box(predictions, biases,
					yolo_n->mask[n], box_index, col, row, yolo_n->w, yolo_n->h,
					netw, neth, yolo_n->w * yolo_n->h);
			dets[count].objectness = objectness;
			dets[count].classes = classes;
			for (j = 0; j < classes; ++j) {
				int class_index = entry_index(yolo_n->w, yolo_n->h,
						n * yolo_n->w * yolo_n->h + i, classes, 4 + 1 + j);
				float prob = objectness * predictions[class_index];
				dets[count].prob[j] = (prob > thresh) ? prob : 0;
			}
			++count;
		}
	}
	correct_yolo_boxes(dets, count, w, h, netw, neth, relative);
	return count;
}

static void fill_network_boxes(yolo_data * post_yolo, int w, int h,
		float thresh, float hier, int *map, int relative, detection *dets) {
	/*This function is calls get_yolo_detections for all yolo layers   */
	int count = get_yolo_detections(post_yolo->yolo_0, w, h, post_yolo->netw,
			post_yolo->neth, thresh, map, relative, dets, post_yolo->no_of_box,
			post_yolo->classes, post_yolo->biases);
	dets += count;
	count = get_yolo_detections(post_yolo->yolo_1, w, h, post_yolo->netw,
			post_yolo->neth, thresh, map, relative, dets, post_yolo->no_of_box,
			post_yolo->classes, post_yolo->biases);
	dets += count;
	count = get_yolo_detections(post_yolo->yolo_2, w, h, post_yolo->netw,
			post_yolo->neth, thresh, map, relative, dets, post_yolo->no_of_box,
			post_yolo->classes, post_yolo->biases);
	dets += count;
}
static detection *get_network_boxes(yolo_data * post_yolo, int w, int h,
		float thresh, float hier, int *map, int relative, int *num) {
	/*
	 w is image w
	 h is image h
	 */
	detection *dets = make_network_boxes(post_yolo, thresh, num);
	fill_network_boxes(post_yolo, w, h, thresh, hier, map, relative, dets);
	return dets;
}

static int nms_comparator(const void *pa, const void *pb) {
	detection a = *(detection *) pa;
	detection b = *(detection *) pb;
	float diff = 0;
	if (b.sort_class >= 0) {
		diff = a.prob[b.sort_class] - b.prob[b.sort_class];
	} else {
		diff = a.objectness - b.objectness;
	}
	if (diff < 0)
		return 1;
	else if (diff > 0)
		return -1;
	return 0;
}
static float overlap(float x1, float w1, float x2, float w2) {
	float l1 = x1 - w1 / 2;
	float l2 = x2 - w2 / 2;
	float left = l1 > l2 ? l1 : l2;
	float r1 = x1 + w1 / 2;
	float r2 = x2 + w2 / 2;
	float right = r1 < r2 ? r1 : r2;
	return right - left;
}
static float box_intersection(box a, box b) {
	float w = overlap(a.x, a.w, b.x, b.w);
	float h = overlap(a.y, a.h, b.y, b.h);
	if (w < 0 || h < 0)
		return 0;
	float area = w * h;
	return area;
}
static float box_union(box a, box b) {
	float i = box_intersection(a, b);
	float u = a.w * a.h + b.w * b.h - i;
	return u;
}
static float box_iou(box a, box b) {
	return box_intersection(a, b) / box_union(a, b);
}
static void do_nms_sort(detection *dets, int total, int classes, float thresh) {
	int i, j, k;
	k = total - 1;
	for (i = 0; i <= k; ++i) {
		if (dets[i].objectness == 0) {
			detection swap = dets[i];
			dets[i] = dets[k];
			dets[k] = swap;
			--k;
			--i;
		}
	}
	total = k + 1;

	for (k = 0; k < classes; ++k) {
		for (i = 0; i < total; ++i) {
			dets[i].sort_class = k;
		}
		qsort(dets, total, sizeof(detection), nms_comparator);
		for (i = 0; i < total; ++i) {
			if (dets[i].prob[k] == 0)
				continue;
			box a = dets[i].bbox;
			for (j = i + 1; j < total; ++j) {
				box b = dets[j].bbox;
				if (box_iou(a, b) > thresh) {
					dets[j].prob[k] = 0;
				}
			}
		}
	}
}

layer_yolo layer_y[3];

/*biase values are anchor values in YOLO cfg file*/
/*There are 18 values total 9 set w and h values got for coco dataset using Kmeans algo*/
/*These values used in Bounding box height and width values*/
float biases[18] = { 10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326 }; //v3

// float biases[18] = { 12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401 }; //v4

/* Mask is to Use last 3 set of anchor values for 1st yolo Layer*/
/*6,7,8, numbers represents 7th,8th,9th set of W ,H values*/
int mask_0[3] = { 6, 7, 8 }; //v3
// int mask_0[3] = { 0, 1, 2 }; //v4
/* Mask is to Use middle 3 set of anchor values for 2nd yolo Layer*/
/*3,4,5, numbers represents 4th,5th,6th set of W ,H values*/
int mask_1[3] = { 3, 4, 5 }; //v3
// int mask_1[3] = { 3, 4, 5 }; //v4

/* Mask is to Use 1st 3 set of anchor values for 3rd yolo Layer*/
/*0,1,2 numbers represents 1st,2nd,3rdh set of W ,H values*/
int mask_2[3] = { 0, 1, 2 }; //v3
// int mask_2[3] = { 6, 7, 8 }; //v4

void init_y(yolo_data *post_data) {
	// initializing with metadata
	post_data->no_of_box = 3;
	post_data->classes = 80;
	post_data->netw = 416;
	post_data->neth = 416;
	post_data->yolo_0 = &layer_y[0];
	post_data->yolo_1 = &layer_y[1];
	post_data->yolo_2 = &layer_y[2];
	post_data->biases = biases;

	/*for 82nd yolo layer*/
	post_data->yolo_0->h = 13;  // 608->19  416->13
	post_data->yolo_0->w = 13;
	post_data->yolo_0->f = 255;
	post_data->yolo_0->mask = mask_0;

	/*for 94th yolo layer*/
	post_data->yolo_1->h = 26;   // 608->38  416->26
	post_data->yolo_1->w = 26;
	post_data->yolo_1->f = 255;
	post_data->yolo_1->mask = mask_1;

	/*for 106th yolo layer*/
	post_data->yolo_2->h = 52;   //608 ->76 416 -> 52
	post_data->yolo_2->w = 52;
	post_data->yolo_2->f = 255;
	post_data->yolo_2->mask = mask_2;

	load_labels();
}

void getdetections(image im, yolo_data *ydata) {
	detection *dets;
	int nboxes = 0;
	forward_yolo(ydata);
	dets = get_network_boxes(ydata, im.w, im.h, 0.5f, 0.5f, 0, 1, &nboxes);
	do_nms_sort(dets, nboxes, ydata->classes, 0.5f); //nms_thresh
	draw_detections(im, dets, nboxes, 0.6f, ydata->classes); //detection_thresh
	free_detections(dets, nboxes);
}