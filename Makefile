TARGET = player

CROSS_COMPILE =


CFLAGS += -I. -I/usr/local/include -g -Wall  -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D_ISOC9X_SOURCE -std=c99

LDFLAGS += -lavdevice  -lavfilter -lswresample -lavformat -lavcodec -lavutil -lswscale -lGLU -lGL -lm -lz -lpthread -lX11 -lSDL


SRC=$(wildcard *.c)
OBJS=${SRC:%.c=%.o}
NAME=${SRC:%.c=%}
DEPS=$(SRC:%.c=.dep/%.d)

.PHONY: dep  all

all: $(OBJS)
	$(CC) -o $(TARGET) $(OBJS) $(LDFLAGS) 
#	$(STRIP) $(TARGET)


clean:
	rm -rf *.o $(TARGET) .dep

%.o: %.c
	${CC} ${CFLAGS} -c $<
	@mkdir -p .dep
	${CC} -MM $(CFLAGS) $*.c > .dep/$*.d 


dep: 
	@mkdir -p .dep
	for i in ${NAME} ; do  \
		${CC} -MM $(CFLAGS) "$${i}".c > .dep/"$${i}".d ;\
	done

