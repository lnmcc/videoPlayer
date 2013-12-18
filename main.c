#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libswresample/swresample.h>
#include <libavutil/avstring.h>
#include <libavutil/pixfmt.h>
#include <libavutil/log.h>
#include <SDL/SDL.h>
#include <SDL/SDL_thread.h>
#include <stdio.h>
#include <math.h>

#define SDL_AUDIO_BUFFER_SIZE 1024
#define MAX_AUDIOQ_SIZE (5 * 16 * 1024)
#define  MAX_VIDEOQ_SIZE (5 * 256 * 1024)
#define VIDEO_PICTURE_QUEUE_SIZE 1

#define FF_ALLOC_EVENT   (SDL_USEREVENT)
#define FF_REFRESH_EVENT (SDL_USEREVENT + 1)
#define FF_QUIT_EVENT (SDL_USEREVENT + 2)

#define SDL_VIDEO_MODE_BPP 24
#define SDL_VIDEO_MODE_FLAGS SDL_HWSURFACE|SDL_RESIZABLE|SDL_ASYNCBLIT|SDL_HWACCEL
#define VIDEO_PICTURE_QUEUE_SIZE 1
#define AV_SYNC_THRESHOLD 0.01
#define AV_NOSYNC_THRESHOLD 10.0
#define SAMPLE_CORRECTION_PERCENT_MAX 10
#define AUDIO_DIFF_AVG_NB 20

#define DEFAULT_AV_SYNC_TYPE AV_SYNC_VIDEO_MASTER

enum {
	AV_SYNC_AUDIO_MASTER,
	AV_SYNC_VIDEO_MASTER,
	AV_SYNC_EXTERNAL_MASTER,
};

typedef struct PacketQueue {
    AVPacketList *first_pkt, *last_pkt;
    int nb_packets;
    int size;
    SDL_mutex *mutex;
    SDL_cond *cond;
} PacketQueue;

typedef struct VideoPicture {
	SDL_Overlay *bmp;
	int width, height;
	int allocated;
	double pts;
} VideoPicture;

typedef struct VideoState {
    char            filename[1024];
    AVFormatContext *ic;
	int				seek_req;
	int				seek_flags;
	int				seek_pos;

   	VideoPicture	pictq[VIDEO_PICTURE_QUEUE_SIZE];
	int				pictq_size, pictq_rindex, pictq_windex;
	SDL_mutex		*pictq_mutex;
	SDL_cond		*pictq_cond;
	int             videoStream;
	AVStream		*video_st;
	PacketQueue		videoq;
	double			video_clock; 

	/* 记录当前播放的帧，在刷新视频的时候会同时更新下面2个参数 */
	/* pts表示的是avcodec的内部时间，而pts_time的单位是微妙，  */
	/* 转换公式:pts_time = pts * TIME_BASE                     */
	int64_t			video_current_pts; 
	/* 当前帧的时间 = av_gettime() */
	int64_t			video_current_pts_time;
	SDL_Thread		*video_tid;


	int				audioStream;
    AVStream        *audio_st;
    AVFrame         *audio_frame;
    PacketQueue     audioq;
    unsigned int    audio_buf_size;
    unsigned int    audio_buf_index;
    AVPacket        audio_pkt;
    uint8_t         *audio_pkt_data;
    int             audio_pkt_size;
    uint8_t         *audio_buf;
    uint8_t         *audio_buf1;
    DECLARE_ALIGNED(16,uint8_t,audio_buf2)[AVCODEC_MAX_AUDIO_FRAME_SIZE * 4];
    enum AVSampleFormat  audio_src_fmt;
    enum AVSampleFormat  audio_tgt_fmt;
    int             audio_src_channels;
    int             audio_tgt_channels;
    int64_t         audio_src_channel_layout;
    int64_t         audio_tgt_channel_layout;
    int             audio_src_freq;
    int             audio_tgt_freq;
	double			audio_clock;
	double			audio_diff_cum;
	double			audio_diff_avg_coef;
	double			audio_diff_threshold;
	double			audio_diff_avg_count;

    struct SwrContext *swr_ctx;
    SDL_Thread      *parse_tid;
    int             quit;
	int				av_sync_type;

	double			frame_timer;
	double			frame_last_pts;
	double			frame_last_delay;
	double			external_clock_base;
} VideoState;

VideoState *global_video_state;
uint64_t global_video_pkt_pts = AV_NOPTS_VALUE;
int g_video_width, g_video_height;
char g_video_resized;
SDL_Surface *screen;
AVPacket flush_pkt;	

void packet_queue_init(PacketQueue *q) {
    memset(q, 0, sizeof(PacketQueue));
    q->mutex = SDL_CreateMutex();
    q->cond = SDL_CreateCond();
}

int packet_queue_put(PacketQueue *q, AVPacket *pkt) {
    AVPacketList *pkt1;

	if(pkt != &flush_pkt && av_dup_packet(pkt) < 0) {
		return -1;
	}

    pkt1 = (AVPacketList *)av_malloc(sizeof(AVPacketList));
    if (!pkt1) {
        return -1;
    }

    pkt1->pkt = *pkt;
    pkt1->next = NULL;

    SDL_LockMutex(q->mutex);

    if (!q->last_pkt) {
        q->first_pkt = pkt1;
    } else {
        q->last_pkt->next = pkt1;
    }

    q->last_pkt = pkt1;
    q->nb_packets++;
    q->size += pkt1->pkt.size;
    SDL_CondSignal(q->cond);
    SDL_UnlockMutex(q->mutex);
    return 0;
}

static int packet_queue_get(PacketQueue *q, AVPacket *pkt, int block) {
    AVPacketList *pkt1;
    int ret;

    SDL_LockMutex(q->mutex);

    for(;;) {
        if(global_video_state->quit) {
            ret = -1;
            break;
        }

        pkt1 = q->first_pkt;
        if (pkt1) {
            q->first_pkt = pkt1->next;
            if (!q->first_pkt) {
                q->last_pkt = NULL;
            }
            q->nb_packets--;
            q->size -= pkt1->pkt.size;
            *pkt = pkt1->pkt;

            av_free(pkt1);
            ret = 1;
            break;
        } else if (!block) {
            ret = 0;
            break;
        } else {
            SDL_CondWait(q->cond, q->mutex);
        }
    }

    SDL_UnlockMutex(q->mutex);

    return ret;
}

static void packet_queue_flush(PacketQueue *q) {
    AVPacketList *pkt, *pkt1;

    SDL_LockMutex(q->mutex);

    for(pkt = q->first_pkt; pkt != NULL; pkt = pkt1) {
        pkt1 = pkt->next;
        av_free_packet(&pkt->pkt);
        av_freep(&pkt);
    }

    q->last_pkt = NULL;
    q->first_pkt = NULL;
    q->nb_packets = 0;
    q->size = 0;

    SDL_UnlockMutex(q->mutex);
}

int audio_decode_frame(VideoState *is, double *pts_ptr) {
    int len1, len2, decoded_data_size;
    AVPacket *pkt = &is->audio_pkt;
    int got_frame = 0;
    int64_t dec_channel_layout;
    int wanted_nb_samples, resampled_data_size;
	int flush_complete = 0;
	double pts = 0;

    for (;;) {
        while (is->audio_pkt_size > 0) {
            if (!is->audio_frame) {
                if (!(is->audio_frame = avcodec_alloc_frame())) {
                    return AVERROR(ENOMEM);
                }
            } else 
                avcodec_get_frame_defaults(is->audio_frame);

			if(flush_complete) {
				break;
			}

            len1 = avcodec_decode_audio4(is->audio_st->codec, is->audio_frame, &got_frame,  pkt);
            if (len1 < 0) {
                // error, skip the frame
                is->audio_pkt_size = 0;
                break;
            }

            is->audio_pkt_data += len1;
            is->audio_pkt_size -= len1;

            if (!got_frame) {
				flush_complete = 1;
                continue;
			}

            decoded_data_size = av_samples_get_buffer_size(NULL,
                                is->audio_frame->channels,
                                is->audio_frame->nb_samples,
                                is->audio_frame->format, 1);

            dec_channel_layout = (is->audio_frame->channel_layout && is->audio_frame->channels
                                  == av_get_channel_layout_nb_channels(is->audio_frame->channel_layout))
                                 ? is->audio_frame->channel_layout
                                 : av_get_default_channel_layout(is->audio_frame->channels);

            wanted_nb_samples =  is->audio_frame->nb_samples;

            if (is->audio_frame->format != is->audio_src_fmt ||
                dec_channel_layout != is->audio_src_channel_layout ||
                is->audio_frame->sample_rate != is->audio_src_freq ||
                (wanted_nb_samples != is->audio_frame->nb_samples && !is->swr_ctx)) {
                if (is->swr_ctx) swr_free(&is->swr_ctx);
                is->swr_ctx = swr_alloc_set_opts(NULL,
                                                 is->audio_tgt_channel_layout,
                                                 is->audio_tgt_fmt,
                                                 is->audio_tgt_freq,
                                                 dec_channel_layout,
                                                 is->audio_frame->format,
                                                 is->audio_frame->sample_rate,
                                                 0, NULL);
                if (!is->swr_ctx || swr_init(is->swr_ctx) < 0) {
                    fprintf(stderr, "swr_init() failed\n");
                    break;
                }
                is->audio_src_channel_layout = dec_channel_layout;
                is->audio_src_channels = is->audio_st->codec->channels;
                is->audio_src_freq = is->audio_st->codec->sample_rate;
                is->audio_src_fmt = is->audio_st->codec->sample_fmt;
            }
            if (is->swr_ctx) {
                const uint8_t **in = (const uint8_t **)is->audio_frame->extended_data; 
                uint8_t *out[] = { is->audio_buf2 };

				if (wanted_nb_samples != is->audio_frame->nb_samples) {
					 if (swr_set_compensation(is->swr_ctx, (wanted_nb_samples - is->audio_frame->nb_samples)
								 * is->audio_tgt_freq / is->audio_frame->sample_rate,
								 wanted_nb_samples * is->audio_tgt_freq / is->audio_frame->sample_rate) < 0) {
						 fprintf(stderr, "swr_set_compensation() failed\n");
						 break;
					 }
				 }

                len2 = swr_convert(is->swr_ctx, out, 
					sizeof(is->audio_buf2) / is->audio_tgt_channels / av_get_bytes_per_sample(is->audio_tgt_fmt),
                    in, is->audio_frame->nb_samples);

                if (len2 < 0) {
                    fprintf(stderr, "swr_convert() failed\n");
                    break;
                }
                if (len2 == sizeof(is->audio_buf2) / is->audio_tgt_channels / av_get_bytes_per_sample(is->audio_tgt_fmt)) {
                    fprintf(stderr, "warning: audio buffer is probably too small\n");
                    swr_init(is->swr_ctx);
                }
                is->audio_buf = is->audio_buf2;
                resampled_data_size = len2 * is->audio_tgt_channels * av_get_bytes_per_sample(is->audio_tgt_fmt);
            } else {
				resampled_data_size = decoded_data_size;
                is->audio_buf = is->audio_frame->data[0];
            }
			pts = is->audio_clock;
			*pts_ptr = pts;
			is->audio_clock += (double)decoded_data_size / (is->audio_st->codec->channels * is->audio_st->codec->sample_rate * av_get_bytes_per_sample(is->audio_st->codec->sample_fmt));
            // We have data, return it and come back for more later
            return resampled_data_size;
        }

        if (pkt->data) av_free_packet(pkt);
		memset(pkt, 0, sizeof(*pkt));
        if (is->quit) 
		  return -1;

        if (packet_queue_get(&is->audioq, pkt, 1) < 0) 
		  return -1;

		if(pkt->data == flush_pkt.data) {
			avcodec_flush_buffers(is->audio_st->codec);
			continue;
		}

        is->audio_pkt_data = pkt->data;
        is->audio_pkt_size = pkt->size;
		if(pkt->pts != AV_NOPTS_VALUE) 
		  is->audio_clock = av_q2d(is->audio_st->codec->time_base) * pkt->pts;
    }
}

int queue_picture(VideoState *is, AVFrame *pFrame, double pts) {
	VideoPicture *vp;
	enum PixelFormat dst_pix_fmt;
	AVPicture pict;
	static struct SwsContext *img_convert_ctx;

	SDL_LockMutex(is->pictq_mutex);
	while(is->pictq_size >= VIDEO_PICTURE_QUEUE_SIZE && !is->quit) {
		SDL_CondWait(is->pictq_cond, is->pictq_mutex);
	}
	SDL_UnlockMutex(is->pictq_mutex);

	if(is->quit) return -1;
	vp = &is->pictq[is->pictq_windex];
	if(!vp->bmp || vp->width != is->video_st->codec->width || vp->height != is->video_st->codec->height || g_video_resized ){
		SDL_Event event;
		vp->allocated = 0;
		event.type = FF_ALLOC_EVENT;
		event.user.data1 = is;
		SDL_PushEvent(&event);
		SDL_LockMutex(is->pictq_mutex);
		while(!vp->allocated && !is->quit) {
			SDL_CondWait(is->pictq_cond, is->pictq_mutex);
		}
		SDL_UnlockMutex(is->pictq_mutex);
		if(is->quit) return -1;
	}
	if(vp->bmp) {
		SDL_LockYUVOverlay(vp->bmp);
		dst_pix_fmt = PIX_FMT_YUV420P;
		pict.data[0] = vp->bmp->pixels[0];
		pict.data[1] = vp->bmp->pixels[2];
		pict.data[2] = vp->bmp->pixels[1];

		pict.linesize[0] = vp->bmp->pitches[0];
		pict.linesize[1] = vp->bmp->pitches[2];
		pict.linesize[2] = vp->bmp->pitches[1];

		if(img_convert_ctx == NULL) {
			int w = is->video_st->codec->width;
			int h = is->video_st->codec->height;
			img_convert_ctx = sws_getContext(w, h, is->video_st->codec->pix_fmt,
											 w, h, dst_pix_fmt, SWS_BICUBIC, 
											 NULL, NULL, NULL);
			if(img_convert_ctx == NULL) {
				fprintf(stderr, "Connot initialize the convertion context!\n");
				exit(1);
			}
		}
		sws_scale(img_convert_ctx, (const uint8_t**)pFrame->data, pFrame->linesize,
									0, is->video_st->codec->height, pict.data, pict.linesize);
		SDL_UnlockYUVOverlay(vp->bmp);
		vp->pts = pts;
		if(++is->pictq_windex == VIDEO_PICTURE_QUEUE_SIZE) is->pictq_windex = 0;
		SDL_LockMutex(is->pictq_mutex);
		is->pictq_size++;
		SDL_UnlockMutex(is->pictq_mutex);
	}
	return 0;
}

/* 获取当前视频时间 */
static double get_video_clock(VideoState *is) {
	double delta;

	delta = (av_gettime() - is->video_current_pts_time) / 1000000.0;
	return is->video_current_pts + delta;
}

/* 获取当前音频时间*/
/* 基本思路是把上一帧的pts加上流逝的时间*/
static double get_audio_clock(VideoState *is) {
	double pts;
	int hw_buf_size, bytes_per_sec, n;

	pts = is->audio_clock;
	hw_buf_size = is->audio_buf_size - is->audio_buf_index;
	bytes_per_sec = 0;
	if(is->audio_st) {
		n = is->audio_st->codec->channels * 2;
		bytes_per_sec = is->audio_st->codec->sample_rate * n;
	}
	if(bytes_per_sec) {
		pts -= (double)hw_buf_size / bytes_per_sec;
	}
	return pts;
}

static double get_external_clock(VideoState *is) {
	return (av_gettime() / 1000000.0) - is->external_clock_base;
}

/* 这个函数根据不同的同步方法来获取当前的播放时间 */
/* 可能是video,audio,local time                   */
static double get_master_clock(VideoState *is) {
	if(is->av_sync_type == AV_SYNC_VIDEO_MASTER) {
		return get_video_clock(is);
	} else if(is->av_sync_type == AV_SYNC_AUDIO_MASTER) {
		return get_audio_clock(is);
	} else {
		return get_external_clock(is);
	}
}

int synchronize_audio(VideoState *is, short *samples, int samples_size, double pts) {
	int n;
	double ref_clock;
	n = 2 * is->audio_st->codec->channels;
	if(is->av_sync_type != AV_SYNC_AUDIO_MASTER) {
		double diff, avg_diff;
		int wanted_size, min_size, max_size, nb_samples;
		ref_clock = get_master_clock(is);
		diff = get_audio_clock(is) - ref_clock;
		if(diff < AV_NOSYNC_THRESHOLD) {
			is->audio_diff_cum = diff + is->audio_diff_avg_coef * is->audio_diff_cum;
			if(is->audio_diff_avg_count < AUDIO_DIFF_AVG_NB) {
				is->audio_diff_avg_count++;
			} else {
				avg_diff = is->audio_diff_cum * (1.0 - is->audio_diff_avg_coef);
				if(fabs(avg_diff) >= is->audio_diff_threshold) {
					wanted_size = samples_size + ((int)(diff * is->audio_st->codec->sample_rate) * n);
					min_size = samples_size * ((100 - SAMPLE_CORRECTION_PERCENT_MAX) / 100);
					max_size = samples_size * ((100 + SAMPLE_CORRECTION_PERCENT_MAX) / 100);
					if(wanted_size < min_size) {
						wanted_size = min_size;
					} else if(wanted_size > max_size) {
						wanted_size = max_size;
					}
					if(wanted_size < samples_size) {
						samples_size = wanted_size;
					} else if(wanted_size > samples_size) {
						uint8_t *samples_end, *q;
						int nb;
						
						nb = (samples_size - wanted_size);
						samples_end = (uint8_t*)samples + samples_size - n;
						q = samples_end + n;
						while(nb > 0) {
							memcpy(q, samples_end, n);
							q += n;
							nb -= n;
						}
						samples_size = wanted_size;
					}
				}
			}
		} else {
			is->audio_diff_avg_count = 0;
			is->audio_diff_cum = 0;
		}
	}
	return samples_size;
}

double synchronize_video(VideoState *is, AVFrame *src_frame, double pts) {
	double frame_delay;

	if(pts != 0) {
		is->video_clock = pts;
	} else {
		pts = is->video_clock;
	}

	frame_delay = av_q2d(is->video_st->codec->time_base);
	frame_delay += src_frame->repeat_pict * (frame_delay * 0.5);
	is->video_clock += frame_delay;
	return pts;
}

int our_get_buffer(struct AVCodecContext *c, AVFrame *pic) {
	int ret = avcodec_default_get_buffer(c, pic);
	uint64_t *pts = (uint64_t*)av_mallocz(sizeof(uint64_t));
	*pts = global_video_pkt_pts;
	pic->opaque = pts;
	//fprintf(stderr, "2: AVFrame.pkt_pts = %lld\n", pic->opaque);
	return ret;
}	

void our_release_buffer(struct AVCodecContext *c, AVFrame *pic) {
	if(pic) av_freep(&pic->opaque);
	avcodec_default_release_buffer(c, pic);
}

void audio_callback(void *userdata, Uint8 *stream, int len) {
    VideoState *is = (VideoState *)userdata;
    int len1, audio_data_size;
	double pts;

    while (len > 0) {
        if (is->audio_buf_index >= is->audio_buf_size) {
            audio_data_size = audio_decode_frame(is, &pts);

            if(audio_data_size < 0) {
                /* silence */
                is->audio_buf_size = 1024;
                memset(is->audio_buf, 0, is->audio_buf_size);
            } else {
                is->audio_buf_size = audio_data_size;
            }
            is->audio_buf_index = 0;
        }

        len1 = is->audio_buf_size - is->audio_buf_index;
        if (len1 > len) {
            len1 = len;
        }

        memcpy(stream, (uint8_t *)is->audio_buf + is->audio_buf_index, len1);
        len -= len1;
        stream += len1;
        is->audio_buf_index += len1;
    }
}

int video_thread(void *arg) {
	VideoState *is = (VideoState*)arg;
	AVPacket pkt1, *packet = &pkt1;
	int len1, frameFinished;
	AVFrame *pFrame;
	double pts;

	pFrame = avcodec_alloc_frame();

	for(;;) {
		if(packet_queue_get(&is->videoq, packet, 1) < 0) {
			fprintf(stderr, "%d: packet_queue_get errror\n", __LINE__);
			break;
		}

		if(packet->data == flush_pkt.data) {
			avcodec_flush_buffers(is->video_st->codec);
			continue;
		}

		pts = 0;
		global_video_pkt_pts = packet->pts;

		len1 = avcodec_decode_video2(is->video_st->codec, pFrame, &frameFinished, packet);
/*
		if(packet->dts == AV_NOPTS_VALUE && *(uint64_t*)pFrame->opaque != AV_NOPTS_VALUE) {
			pts = *(uint64_t*)pFrame->opaque;
		} else if(packet->dts != AV_NOPTS_VALUE) {
			pts = packet->dts;
		} else {
			pts = 0;
		}

		pts *= av_q2d(is->video_st->time_base);
*/
		if(frameFinished) {
			pts = synchronize_video(is, pFrame, pts);
			if(queue_picture(is, pFrame, pts) < 0) 
			  break;
		}
		av_free_packet(packet); }
	av_free(pFrame);
	return 0;
}

int stream_component_open(VideoState *is, int stream_index) {
		AVFormatContext *ic = is->ic;
		AVCodecContext *codecCtx;
		AVCodec *codec;
		SDL_AudioSpec wanted_spec, spec;
		int64_t wanted_channel_layout = 0;
		int wanted_nb_channels;
		const int next_nb_channels[] = {0, 0, 1 ,6, 2, 6, 4, 6};

		if (stream_index < 0 || stream_index >= ic->nb_streams) {
			return -1;
		}
		
		codecCtx = ic->streams[stream_index]->codec;

		if(codecCtx->codec_type == AVMEDIA_TYPE_AUDIO) {
		wanted_nb_channels = codecCtx->channels;
		if(!wanted_channel_layout || wanted_nb_channels != av_get_channel_layout_nb_channels(wanted_channel_layout)) {
			wanted_channel_layout = av_get_default_channel_layout(wanted_nb_channels);
			wanted_channel_layout &= ~AV_CH_LAYOUT_STEREO_DOWNMIX;
		}
		
		wanted_spec.channels = av_get_channel_layout_nb_channels(wanted_channel_layout);
		wanted_spec.freq = codecCtx->sample_rate;
		if (wanted_spec.freq <= 0 || wanted_spec.channels <= 0) {
			fprintf(stderr, "Invalid sample rate or channel count!\n");
			return -1;
		}
	}

	if(codecCtx->codec_type == AVMEDIA_TYPE_AUDIO) {
		wanted_spec.format = AUDIO_S16SYS;
		wanted_spec.silence = 0;
		wanted_spec.samples = SDL_AUDIO_BUFFER_SIZE;
		wanted_spec.callback = audio_callback;
		wanted_spec.userdata = is;
		
		while(SDL_OpenAudio(&wanted_spec, &spec) < 0) {
			fprintf(stderr, "SDL_OpenAudio (%d channels): %s\n", wanted_spec.channels, SDL_GetError());
			wanted_spec.channels = next_nb_channels[FFMIN(7, wanted_spec.channels)];
			if(!wanted_spec.channels) {
				fprintf(stderr, "No more channel combinations to tyu, audio open failed\n");
				return -1;
			}
			wanted_channel_layout = av_get_default_channel_layout(wanted_spec.channels);
		}

		if (spec.format != AUDIO_S16SYS) {
			fprintf(stderr, "SDL advised audio format %d is not supported!\n", spec.format);
			return -1;
		}
		if (spec.channels != wanted_spec.channels) {
			wanted_channel_layout = av_get_default_channel_layout(spec.channels);
			if (!wanted_channel_layout) {
				fprintf(stderr, "SDL advised channel count %d is not supported!\n", spec.channels);
				return -1;
			}
		}

		fprintf(stderr, "%d: wanted_spec.format = %d\n", __LINE__, wanted_spec.format);
		fprintf(stderr, "%d: wanted_spec.samples = %d\n", __LINE__, wanted_spec.samples);
		fprintf(stderr, "%d: wanted_spec.channels = %d\n", __LINE__, wanted_spec.channels);
		fprintf(stderr, "%d: wanted_spec.freq = %d\n", __LINE__, wanted_spec.freq);

		fprintf(stderr, "%d: spec.format = %d\n", __LINE__, spec.format);
		fprintf(stderr, "%d: spec.samples = %d\n", __LINE__, spec.samples);
		fprintf(stderr, "%d: spec.channels = %d\n", __LINE__, spec.channels);
		fprintf(stderr, "%d: spec.freq = %d\n", __LINE__, spec.freq);

		is->audio_src_fmt = is->audio_tgt_fmt = AV_SAMPLE_FMT_S16;
		is->audio_src_freq = is->audio_tgt_freq = spec.freq;
		is->audio_src_channel_layout = is->audio_tgt_channel_layout = wanted_channel_layout;
		is->audio_src_channels = is->audio_tgt_channels = spec.channels;
	}	

    codec = avcodec_find_decoder(codecCtx->codec_id);
    if (!codec || (avcodec_open2(codecCtx, codec, NULL) < 0)) {
        fprintf(stderr, "Unsupported codec!\n");
        return -1;
    }

//	ic->streams[stream_index]->discard = AVDISCARD_DEFAULT;
    switch(codecCtx->codec_type) {
    case AVMEDIA_TYPE_AUDIO:
        is->audioStream = stream_index;
        is->audio_st = ic->streams[stream_index];
        is->audio_buf_size = 0;
        is->audio_buf_index = 0;
        memset(&is->audio_pkt, 0, sizeof(is->audio_pkt));
        packet_queue_init(&is->audioq);
        SDL_PauseAudio(0);
        break;
	case AVMEDIA_TYPE_VIDEO:
		is->videoStream = stream_index;
		is->video_st = ic->streams[stream_index];
		is->frame_timer = (double)av_gettime() / 1000000.0;
		is->frame_last_delay = 40e-3;
		is->video_current_pts_time = av_gettime();
		packet_queue_init(&is->videoq);
		is->video_tid = SDL_CreateThread(video_thread, is);
		codecCtx->get_buffer = our_get_buffer;
		codecCtx->release_buffer = our_release_buffer;
    default:
        break;
    }
}



static int decode_interrupt_cb(void *arg) {
	return (global_video_state && global_video_state->quit);
}

static int decode_thread(void *arg) {
    VideoState *is = (VideoState *)arg;
    AVFormatContext *ic = NULL;
    AVPacket pkt1, *packet = &pkt1;
    int ret, i;
	int video_index = -1;
	int audio_index = -1;

    is->audioStream = -1;
	is->videoStream = -1;

    global_video_state = is;
    if (avformat_open_input(&ic, is->filename, NULL, NULL) != 0) {
        return -1;
    }

    /* 这个回调函数将赋值给AVFormatContex,这样当读取流出现问题的时候会调用我们的自己的处理 */
	static const AVIOInterruptCB int_cb = { decode_interrupt_cb, NULL };
	ic->interrupt_callback = int_cb;

    is->ic = ic;
	is->external_clock_base = 0;
	is->external_clock_base = get_external_clock(is);

    if (avformat_find_stream_info(ic, NULL) < 0) {
        return -1;
    }

    av_dump_format(ic, 0, is->filename, 0);

    for(i=0; i<ic->nb_streams; i++) {
        if(ic->streams[i]->codec->codec_type == AVMEDIA_TYPE_AUDIO && audio_index < 0) 
            audio_index = i;
		if(ic->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO && video_index < 0)
			video_index = i;
    }

    if(audio_index >= 0) stream_component_open(is, audio_index);
	
	if(video_index >= 0) stream_component_open(is, video_index); 

	//TODO:
/*    if (is->audioStream < 0 || is->videoStream < 0) {
        fprintf(stderr, "%s: could not open codecs\n", is->filename);
        goto fail;
    }
	*/
    /* 开始解码主循环 */
    for(;;) {
        if(is->quit) 
		  break;
		/* 这里处理视频的快进和快退 */
		if(is->seek_req) {
			int stream_index = -1;
			int64_t seek_target = is->seek_pos;

			is->external_clock_base = 0;
			is->external_clock_base = get_external_clock(is) - (seek_target / 1000000.0);
			
			if(is->videoStream >= 0) 
			  stream_index = is->videoStream;
			else if(is->audioStream >= 0)
					stream_index = is->audioStream;
		
			/* av_rescale_q(a, b, c):通过计算a*b/c来把一个时间基调整到另一个时间基 */
			/* 使用这个函数的原因是可以防止计算溢出，AV_TIME_BASE_Q是AV_TIME_BASE作*/
			/* 为分母的版本                                                        */
			if(stream_index >= 0) 
				seek_target = av_rescale_q(seek_target, AV_TIME_BASE_Q , ic->streams[stream_index]->time_base);

			if(av_seek_frame(is->ic, stream_index, seek_target, is->seek_flags) < 0) {
				fprintf(stderr, "%d: %s error seek\n", __LINE__, is->filename);
			} else {
				/* 跳转后需要清空我们自己的缓冲队列和avcodec内部缓冲*/
				/* 然后放入一个用来标识刷新队列的标志包             */
				if(is->audioStream >= 0) {
					packet_queue_flush(&is->audioq);
					packet_queue_put(&is->audioq, &flush_pkt);
				}
				if(is->videoStream >= 0) {
					packet_queue_flush(&is->videoq);
					packet_queue_put(&is->videoq, &flush_pkt);
				}
			}

			is->seek_req = 0;
		}

        if (is->audioq.size > MAX_AUDIOQ_SIZE || is->videoq.size > MAX_VIDEOQ_SIZE) {
            SDL_Delay(10);
            continue;
        }

        ret = av_read_frame(is->ic, packet);
        if (ret < 0) {
            if(ret == AVERROR_EOF || url_feof(is->ic->pb)) {
                break;
            }
            if(is->ic->pb && is->ic->pb->error) {
                break;
            }
            continue;
        }

		if(packet->stream_index == is->videoStream) {
			packet_queue_put(&is->videoq, packet);
		} else if(packet->stream_index == is->audioStream) {
            packet_queue_put(&is->audioq, packet);
        } else {
            av_free_packet(packet);
        }
    }

    while (!is->quit) {
        SDL_Delay(100);
    }

fail: {
        SDL_Event event;
        event.type = FF_QUIT_EVENT;
        event.user.data1 = is;
        SDL_PushEvent(&event);
    }

    return 0;
}

void alloc_picture(void *userdata) {
	VideoState *is = (VideoState*)userdata;
	VideoPicture *vp;

	vp = &is->pictq[is->pictq_windex];
	if(vp->bmp) {
		SDL_FreeYUVOverlay(vp->bmp);
	}

	if(g_video_resized) {
		screen = NULL;
		screen = SDL_SetVideoMode(g_video_width, g_video_height, SDL_VIDEO_MODE_BPP, SDL_VIDEO_MODE_FLAGS);
		g_video_resized = 0;
	}
	vp->bmp = SDL_CreateYUVOverlay(is->video_st->codec->width,
								   is->video_st->codec->height,
								   SDL_YV12_OVERLAY,
								   screen);

	vp->width = is->video_st->codec->width;
	vp->height = is->video_st->codec->height;

	SDL_LockMutex(is->pictq_mutex);
	vp->allocated = 1;
	SDL_CondSignal(is->pictq_cond);
	SDL_UnlockMutex(is->pictq_mutex);
}

static Uint32 sdl_refresh_timer_cb(Uint32 interval, void *opaque) {
	SDL_Event event;
	event.type = FF_REFRESH_EVENT;
	event.user.data1 = opaque;
	SDL_PushEvent(&event);

	return 0; // 0 means stop timer
}

/* delay毫秒刷新 */
static void schedule_refresh(VideoState *is, int delay) {
	SDL_AddTimer(delay, sdl_refresh_timer_cb, is);
}

void video_display(VideoState *is) {
	SDL_Rect rect;
	VideoPicture *vp;
	AVPicture pict;
	float aspect_ratio;
	int w, h, x, y;
	int i;

	vp = &is->pictq[is->pictq_rindex];
	if(vp->bmp) {
		if(is->video_st->codec->sample_aspect_ratio.num == 0) {
			aspect_ratio = 0;
		} else {
			aspect_ratio = av_q2d(is->video_st->codec->sample_aspect_ratio) *
								  is->video_st->codec->width / is->video_st->codec->height;
		}
		if(aspect_ratio <= 0.0) 
			aspect_ratio = (float)is->video_st->codec->width / (float)is->video_st->codec->height;
		
		h = screen->h;
		w = ((int)(h * aspect_ratio)) & -3;
		if(w > screen->w) {
			w = screen->w;
			h = ((int)(w / aspect_ratio)) & -3;
		}
		x = (screen->w - w) / 2;
		y = (screen->h - h) / 2;

		rect.x = x;
		rect.y = y;
		rect.w = w;
		rect.h = h;

		SDL_DisplayYUVOverlay(vp->bmp, &rect);
	}
}

void stream_seek(VideoState *is, int64_t pos, int rel) {
	if(!is->seek_req) {
		is->seek_pos = pos;
		is->seek_flags = rel < 0 ? AVSEEK_FLAG_BACKWARD : 0;
		is->seek_req = 1;
	}
}

void video_refresh_timer(void *userdata) {
	VideoState *is = (VideoState*)userdata;
	VideoPicture *vp;
	double actual_delay, delay, sync_threshold, ref_clock, diff;

	if(is->video_st) {
		if(is->pictq_size == 0) {
			schedule_refresh(is, 1);		
		} else {
			vp = &is->pictq[is->pictq_rindex];
			is->video_current_pts = vp->pts;
			is->video_current_pts_time = av_gettime();
			delay = vp->pts - is->frame_last_pts;
			if(delay <= 0 || delay >= 1.0) delay = is->frame_last_delay;
			is->frame_last_delay = delay;
			is->frame_last_pts = vp->pts;
			if(is->av_sync_type != AV_SYNC_VIDEO_MASTER) {
				ref_clock = get_master_clock(is);
				diff = vp->pts - ref_clock;
				sync_threshold = (delay > AV_SYNC_THRESHOLD) ? delay : AV_SYNC_THRESHOLD;
				if(fabs(diff) < AV_NOSYNC_THRESHOLD) {
					if(diff <= -sync_threshold)	{
						delay = 0;
					} else if(diff >= sync_threshold) {
						delay = 2 * delay;
					}
				}
			}
			is->frame_timer += delay;	
			actual_delay = is->frame_timer - (av_gettime() / 1000000.0);
			if(actual_delay < 0.010) actual_delay = 0.010;
			schedule_refresh(is, (int)(actual_delay * 1000 + 0.5));
			video_display(is);
			if(++is->pictq_rindex == VIDEO_PICTURE_QUEUE_SIZE) is->pictq_rindex = 0;
			SDL_LockMutex(is->pictq_mutex);
			is->pictq_size--;
			SDL_CondSignal(is->pictq_cond);
			SDL_UnlockMutex(is->pictq_mutex);
		 }
	} else
	  schedule_refresh(is, 100);
}

int main(int argc, char *argv[]) {
    SDL_Event       event;
	double			pos;
    VideoState      *is;


    if(argc < 2) {
        fprintf(stderr, "Usage: test <file>\n");
        exit(1);
    }

	XInitThreads();

    is = (VideoState *)av_mallocz(sizeof(VideoState));

	avcodec_register_all();
	avdevice_register_all();
	avfilter_register_all();
    av_register_all();
	avformat_network_init();

    if(SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO | SDL_INIT_TIMER)) {
        fprintf(stderr, "Could not initialize SDL - %s\n", SDL_GetError());
        exit(1);
    }

	g_video_width = 640;
	g_video_height = 480;
	g_video_resized = 0;

	screen = SDL_SetVideoMode(g_video_width, g_video_height, SDL_VIDEO_MODE_BPP, SDL_VIDEO_MODE_FLAGS);
	if(!screen) {
		fprintf(stderr, "SDL: could not set video mode - exiting\n");
		exit(1);
	}

    av_strlcpy(is->filename, argv[1], sizeof(is->filename));

	is->pictq_mutex = SDL_CreateMutex();
	is->pictq_cond = SDL_CreateCond();

	schedule_refresh(is, 40);

	is->av_sync_type = AV_SYNC_VIDEO_MASTER;

    is->parse_tid = SDL_CreateThread(decode_thread, is);
    if (!is->parse_tid) {
        av_free(is);
        return -1;
    }

	av_init_packet(&flush_pkt);
	flush_pkt.data = (uint8_t*)"FLUSH";

    for(;;) {
		double incr, pos;

        SDL_WaitEvent(&event);
        switch(event.type) {
		case SDL_VIDEORESIZE:
			g_video_width = event.resize.w;
			g_video_height = event.resize.h;
			g_video_resized = 1;
			break;
		case SDL_KEYDOWN:
			switch(event.key.keysym.sym) {
			case SDLK_LEFT:
				incr = -1.0;
				goto do_seek;
			case SDLK_RIGHT:
				incr = 1.0;
				goto do_seek;
			case SDLK_UP:
				incr = 6.0;
				goto do_seek;
			case SDLK_DOWN:
				incr = -6.0;
				goto do_seek;
			do_seek:
				if(global_video_state) {
					/* 获取当前播放位置 */
					pos = get_master_clock(global_video_state);
					pos += incr;
					stream_seek(global_video_state, (int64_t)(pos * AV_TIME_BASE), incr);
				}
			default: 
				break;
			}
			break;
        case FF_QUIT_EVENT:
        case SDL_QUIT:
            is->quit = 1;
            SDL_Quit();
            exit(0);
            break;
		case FF_ALLOC_EVENT:
			alloc_picture(event.user.data1);
			break;
		case FF_REFRESH_EVENT:
			video_refresh_timer(event.user.data1);
        default:
            break;
        }
    }
    return 0;
}
