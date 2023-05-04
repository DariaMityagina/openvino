// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string.h>
#include "stdlib.h"

#include "XLinkMacros.h"
#include "XLinkErrorUtils.h"
#include "XLinkPlatform.h"
#include "XLinkDispatcherImpl.h"
#include "XLinkPrivateFields.h"

#ifdef MVLOG_UNIT_NAME
#undef MVLOG_UNIT_NAME
#define MVLOG_UNIT_NAME xLink
#endif
#include "XLinkLog.h"
#include "XLinkStringUtils.h"
#include <time.h>
#include <stdio.h>
// ------------------------------------
// Helpers declaration. Begin.
// ------------------------------------

static int isStreamSpaceEnoughFor(streamDesc_t* stream, uint32_t size);

static streamPacketDesc_t* getPacketFromStream(streamDesc_t* stream);
static int releasePacketFromStream(streamDesc_t* stream, uint32_t* releasedSize, FILE *df);
static int releaseSpecificPacketFromStream(streamDesc_t* stream, uint32_t* releasedSize, uint8_t* data, FILE *df);
static int addNewPacketToStream(streamDesc_t* stream, void* buffer, uint32_t size, FILE *df);

static int handleIncomingEvent(xLinkEvent_t* event, FILE *df);

// ------------------------------------
// Helpers declaration. End.
// ------------------------------------

extern FILE *globalDebugFile;

// ------------------------------------
// XLinkDispatcherImpl.h implementation. Begin.
// ------------------------------------
#define USB_SIZE_MULTIPLE (1024)

char* T2S(int type)
{
    switch(type)
    {
        case XLINK_WRITE_REQ:           return "WRITE_REQ         ";
        case XLINK_READ_REQ:            return "READ_REQ          ";
        case XLINK_READ_REL_REQ:        return "READ_REL_REQ      ";
        case XLINK_READ_REL_SPEC_REQ:   return "READ_REL_SPEC_REQ ";
        case XLINK_CREATE_STREAM_REQ:   return "CREATE_STREAM_REQ ";
        case XLINK_CLOSE_STREAM_REQ:    return "CLOSE_STREAM_REQ  ";
        case XLINK_PING_REQ:            return "PING_REQ          ";
        case XLINK_RESET_REQ:           return "RESET_REQ         ";
        case XLINK_REQUEST_LAST:        return "REQUEST_LAST      ";
        case XLINK_WRITE_RESP:          return "WRITE_RESP        ";
        case XLINK_READ_RESP:           return "READ_RESP         ";
        case XLINK_READ_REL_RESP:       return "READ_REL_RESP     ";
        case XLINK_READ_REL_SPEC_RESP:  return "READ_REL_SPEC_RESP";
        case XLINK_CREATE_STREAM_RESP:  return "CREATE_STREAM_RESP";
        case XLINK_CLOSE_STREAM_RESP:   return "CLOSE_STREAM_RESP ";
        case XLINK_PING_RESP:           return "PING_RESP         ";
        case XLINK_RESET_RESP:          return "RESET_RESP        ";
        case XLINK_RESP_LAST:           return "RESP_LAST         ";
    }
    return "unknown           ";
}

static uint32_t FD_input_count = 0;
static uint32_t FD_output_count = 0;
static uint32_t ASD_input_count = 0;
static uint32_t ASD_ouput_count = 0;

//adds a new event with parameters and returns event id
int dispatcherEventSend(xLinkEvent_t *event, FILE *df)
{
    mvLog(MVLOG_DEBUG, "Send event: %s, size %d, streamId %ld.\n",
        TypeToStr(event->header.type), event->header.size, event->header.streamId);

    struct timespec ts; //timespec_get(&ts, TIME_UTC);
    char buff[300]; char timeStamp[300];

    int rc = XLinkPlatformWrite(&event->deviceHandle,
        &event->header, sizeof(event->header));

    clock_gettime(CLOCK_REALTIME, &ts);
    strftime(buff, sizeof buff, "%T", gmtime(&ts.tv_sec));
    sprintf(timeStamp, "%s.%06ld", buff, ts.tv_nsec / 1000);

    uint32_t devhnd = (uint32_t)(event->deviceHandle.xLinkFD) & 0xFFFFF;
    
    if(rc < 0) {
        mvLog(MVLOG_ERROR,"Write failed (header) (err %d) | event %s\n", rc, TypeToStr(event->header.type));
        // fprintf(globalDebugFile, "%s XLink header error %d %x %s ch_%d id_%d %d\n", timeStamp, rc, devhnd, T2S(event->header.type), event->header.streamId, event->header.id, event->header.size); fflush(globalDebugFile);
        return rc;
    }

    if (event->header.type == XLINK_WRITE_REQ) {
        rc = XLinkPlatformWrite(&event->deviceHandle,
            event->data, event->header.size);

        if(rc < 0) {
            // fprintf(globalDebugFile, "%s XLink data error %d %x %s ch_%d id_%d %d\n", timeStamp, rc, devhnd, T2S(event->header.type), event->header.streamId, event->header.id, event->header.size); fflush(globalDebugFile);
            mvLog(MVLOG_ERROR,"Write failed %d\n", rc);
            return rc;
        }

        if (event->header.streamId == 0) {
            //graph monitor
            switch(((uint8_t*)event->data)[0]) {
                case 0:  fprintf(globalDebugFile, "%s S+ %x GRAPH_ALLOCATE_CMD         \n", timeStamp, devhnd); fflush(globalDebugFile); break;
                case 1:  fprintf(globalDebugFile, "%s S+ %x GRAPH_DEALLOCATE_CMD       \n", timeStamp, devhnd); fflush(globalDebugFile); break;
                case 2:  fprintf(globalDebugFile, "%s S+ %x GRAPH_TRIGGER_CMD          ch_%d id_%d s_%d\n", timeStamp, devhnd, event->header.streamId, event->header.id, event->header.size); fflush(globalDebugFile); break;
                case 3:  fprintf(globalDebugFile, "%s S+ %x GRAPH_VERIFY_CMD           \n", timeStamp, devhnd); fflush(globalDebugFile); break;
                case 4:  fprintf(globalDebugFile, "%s S+ %x GRAPH_ALLOCATION_VERIFY_CMD\n", timeStamp, devhnd); fflush(globalDebugFile); break;
                case 5:  fprintf(globalDebugFile, "%s S+ %x GRAPH_BUFFER_ALLOCATE_CMD  \n", timeStamp, devhnd); fflush(globalDebugFile); break;
                case 6:  fprintf(globalDebugFile, "%s S+ %x GRAPH_BUFFER_DEALLOCATE_CMD\n", timeStamp, devhnd); fflush(globalDebugFile); break;
                case 7:  fprintf(globalDebugFile, "%s S+ %x GRAPH_GET_TIMING_DATA      \n", timeStamp, devhnd); fflush(globalDebugFile); break;
                case 8:  fprintf(globalDebugFile, "%s S+ %x GRAPH_GET_DEBUG_DATA       \n", timeStamp, devhnd); fflush(globalDebugFile); break;
                case 10: fprintf(globalDebugFile, "%s S+ %x GRAPH_COMMAND_LAST         \n", timeStamp, devhnd); fflush(globalDebugFile); break;
                default: fprintf(globalDebugFile, "%s S+ %x unknown for graph monitor  \n", timeStamp, devhnd); fflush(globalDebugFile); break;
            }
        } else if (event->header.streamId == 1) {
            //device monitor
            switch(((uint8_t*)event->data)[0]) {
                case   0:  fprintf(globalDebugFile, "%s S+ %x DEVICE_GET_THERMAL_STATS        \n", timeStamp, devhnd); fflush(globalDebugFile); break;
                case   1:  fprintf(globalDebugFile, "%s S+ %x DEVICE_GET_CAPABILITIES         \n", timeStamp, devhnd); fflush(globalDebugFile); break;
                case   2:  fprintf(globalDebugFile, "%s S+ %x DEVICE_GET_USED_MEMORY          \n", timeStamp, devhnd); fflush(globalDebugFile); break;
                case   3:  fprintf(globalDebugFile, "%s S+ %x DEVICE_GET_DEVICE_ID            \n", timeStamp, devhnd); fflush(globalDebugFile); break;
                case   4:  fprintf(globalDebugFile, "%s S+ %x DEVICE_WATCHDOG_PING            ch_%d id_%d s_%d\n", timeStamp, devhnd, event->header.streamId, event->header.id, event->header.size); fflush(globalDebugFile); break;
                case   5:  fprintf(globalDebugFile, "%s S+ %x DEVICE_SET_STDIO_REDIRECT_XLINK \n", timeStamp, devhnd); fflush(globalDebugFile); break;
                case   6:  fprintf(globalDebugFile, "%s S+ %x DEVICE_SET_POWER_CONFIG         \n", timeStamp, devhnd); fflush(globalDebugFile); break;
                case   7:  fprintf(globalDebugFile, "%s S+ %x DEVICE_RESET_POWER_CONFIG       \n", timeStamp, devhnd); fflush(globalDebugFile); break;
                case   8:  fprintf(globalDebugFile, "%s S+ %x DEVICE_ENABLE_ASYNC_DMA         \n", timeStamp, devhnd); fflush(globalDebugFile); break;
                case   9:  fprintf(globalDebugFile, "%s S+ %x DEVICE_COMMAND_LAST             \n", timeStamp, devhnd); fflush(globalDebugFile); break;
                default :  fprintf(globalDebugFile, "%s S+ %x unknown for device monitor      \n", timeStamp, devhnd); fflush(globalDebugFile); break;
            }
        } else if (event->header.streamId == 2) {
            if (event->header.size == 132) {
                fprintf(globalDebugFile, "%s S+ %x network blob header\n", timeStamp, devhnd); fflush(globalDebugFile);
            } else if (event->header.size == 1867648) {
                fprintf(globalDebugFile, "%s S+ %x ASD network blob\n", timeStamp, devhnd); fflush(globalDebugFile);
            } else if (event->header.size == 842304) {
                fprintf(globalDebugFile, "%s S+ %x FD network blob\n", timeStamp, devhnd); fflush(globalDebugFile);
            }
        }
        else if (event->header.streamId == 3) {
            if (event->header.size == 602112) {
                fprintf(globalDebugFile, "%s S+ %x FD input #%u ch_%d id_%d s_%d\n", timeStamp, devhnd, FD_input_count++, event->header.streamId, event->header.id, event->header.size); fflush(globalDebugFile);
            }
            if (event->header.size == 450688) {
                fprintf(globalDebugFile, "%s S+ %x ASD input #%u ch_%d id_%d s_%d\n", timeStamp, devhnd, ASD_input_count++, event->header.streamId, event->header.id, event->header.size); fflush(globalDebugFile);
            }
        }
    }
    else {
        fprintf(globalDebugFile, "%s S  %x %s ch_%d id_%d s_%d\n"   , timeStamp, devhnd, T2S(event->header.type), event->header.streamId, event->header.id, event->header.size); fflush(globalDebugFile);
        // // fprintf(globalDebugFile, "%s S  %x ---\n"   , timeStamp, devhnd); fflush(globalDebugFile);
        //     switch(event->header.type) {
        //         case XLINK_READ_REQ:           fprintf(globalDebugFile, "%s S  %x %s ch_%d id_%d\n"   , timeStamp, devhnd, T2S(event->header.type), event->header.streamId, event->header.id); fflush(globalDebugFile); break;
        //         case XLINK_READ_REL_REQ:       fprintf(globalDebugFile, "%s S  %x %s ch_%d id_%d\n"   , timeStamp, devhnd, T2S(event->header.type), event->header.streamId, event->header.id); fflush(globalDebugFile); break;
        //         case XLINK_READ_REL_SPEC_REQ:  fprintf(globalDebugFile, "%s S  %x %s ch_%d id_%d\n"   , timeStamp, devhnd, T2S(event->header.type), event->header.streamId, event->header.id); fflush(globalDebugFile); break;
        //         case XLINK_CREATE_STREAM_REQ:  fprintf(globalDebugFile, "%s S  %x %s id_%d %s\n"      , timeStamp, devhnd, T2S(event->header.type), event->header.id, event->header.streamName); fflush(globalDebugFile); break;
        //         case XLINK_CLOSE_STREAM_REQ:   fprintf(globalDebugFile, "%s S  %x %s ch_%d id_%d %s\n", timeStamp, devhnd, T2S(event->header.type), event->header.streamId, event->header.id, event->header.streamName); fflush(globalDebugFile); break;
        //         case XLINK_PING_REQ:           fprintf(globalDebugFile, "%s S  %x %s id_%d \n"        , timeStamp, devhnd, T2S(event->header.type), event->header.id); fflush(globalDebugFile); break;
        //         case XLINK_RESET_REQ:          fprintf(globalDebugFile, "%s S  %x %s ch_%d id_%d \n"  , timeStamp, devhnd, T2S(event->header.type), event->header.streamId, event->header.id); fflush(globalDebugFile); break;
        //         case XLINK_REQUEST_LAST:       fprintf(globalDebugFile, "%s S  %x %s ch_%d id_%d \n"  , timeStamp, devhnd, T2S(event->header.type), event->header.streamId, event->header.id); fflush(globalDebugFile); break;
        //         case XLINK_WRITE_RESP:         fprintf(globalDebugFile, "%s S  %x %s ch_%d id_%d \n"  , timeStamp, devhnd, T2S(event->header.type), event->header.streamId, event->header.id); fflush(globalDebugFile); break;
        //         case XLINK_READ_RESP:          fprintf(globalDebugFile, "%s S  %x %s ch_%d id_%d \n"  , timeStamp, devhnd, T2S(event->header.type), event->header.streamId, event->header.id); fflush(globalDebugFile); break;
        //         case XLINK_READ_REL_RESP:      fprintf(globalDebugFile, "%s S  %x %s ch_%d id_%d \n"  , timeStamp, devhnd, T2S(event->header.type), event->header.streamId, event->header.id); fflush(globalDebugFile); break;
        //         case XLINK_READ_REL_SPEC_RESP: fprintf(globalDebugFile, "%s S  %x %s ch_%d id_%d \n"  , timeStamp, devhnd, T2S(event->header.type), event->header.streamId, event->header.id); fflush(globalDebugFile); break;
        //         case XLINK_CREATE_STREAM_RESP: fprintf(globalDebugFile, "%s S  %x %s id_%d %s\n"      , timeStamp, devhnd, T2S(event->header.type), event->header.id, event->header.streamName); fflush(globalDebugFile); break;
        //         case XLINK_CLOSE_STREAM_RESP:  fprintf(globalDebugFile, "%s S  %x %s id_%d %s\n"      , timeStamp, devhnd, T2S(event->header.type), event->header.id, event->header.streamName); fflush(globalDebugFile); break;
        //         case XLINK_PING_RESP:          fprintf(globalDebugFile, "%s S  %x %s id_%d \n"        , timeStamp, devhnd, T2S(event->header.type), event->header.id); fflush(globalDebugFile); break;
        //         case XLINK_RESET_RESP:         fprintf(globalDebugFile, "%s S  %x %s ch_%d id_%d \n",   timeStamp, devhnd, T2S(event->header.type), event->header.streamId, event->header.id); fflush(globalDebugFile); break;
        //         case XLINK_RESP_LAST:          fprintf(globalDebugFile, "%s S  %x %s ch_%d id_%d \n",   timeStamp, devhnd, T2S(event->header.type), event->header.streamId, event->header.id); fflush(globalDebugFile); break;
        //     };
    }

    return 0;
}

int dispatcherEventReceive(xLinkEvent_t* event, xLinkEvent_t* prevEvent, FILE *df){
    // uint8_t headerRxBuffer[USB_SIZE_MULTIPLE];
    int rc = 0;
	uint32_t devhnd = (uint32_t)(event->deviceHandle.xLinkFD) & 0xFFFFF;
    int untilNotDebugMessage = 1;
	char timeStamp[300];
    uint32_t blopCount = 0;

    do {
        // rc = XLinkPlatformRead(&event->deviceHandle, headerRxBuffer, sizeof(event->header));
		// memcpy(&event->header, headerRxBuffer, sizeof(event->header));
        rc = XLinkPlatformRead(&event->deviceHandle, &event->header, sizeof(event->header));

        struct timespec ts; //timespec_get(&ts, TIME_UTC);
        char buff[300];
        clock_gettime(CLOCK_REALTIME, &ts);
        strftime(buff, sizeof buff, "%T", gmtime(&ts.tv_sec));
        sprintf(timeStamp, "%s.%06ld", buff, ts.tv_nsec / 1000);

        untilNotDebugMessage = 0;
        if (rc == 0) {
            // fprintf(globalDebugFile, "%s\t\t\t\tSideChannelStart %d %x\n", timeStamp, event->header.type, devhnd); fflush(globalDebugFile);
            if (event->header.type == 333) { //debug string
                untilNotDebugMessage = 1;
                char mm[1024] = {0};
                rc = XLinkPlatformRead(&event->deviceHandle, mm, sizeof(mm));
                if (rc == 0) {
                    if (event->sideChanneldebugFile) {
                        fprintf(event->sideChanneldebugFile, "%s %s\n", timeStamp, mm); fflush(event->sideChanneldebugFile);
                    }
                    // fprintf(globalDebugFile, "%s %x - %s\n", timeStamp, devhnd, mm); fflush(globalDebugFile);
                    // fprintf(df, "%s %x - %s\n", timeStamp, devhnd, mm); fflush(df);
                }
                else {
                    // fprintf(globalDebugFile, "%s\t\t\t\tSideChannel error1 %d %x %d\n", timeStamp, event->header.type, devhnd, rc); fflush(globalDebugFile);
                }
            }
            else if (event->header.type == 444) { //debug binary data
                untilNotDebugMessage = 1;
                uint8_t *mm = (uint8_t*)malloc(event->header.size);
                rc = XLinkPlatformRead(&event->deviceHandle, mm, sizeof(mm));
                if (rc == 0) {
                    if (event->sideChanneldebugFile) {
                        fprintf(event->sideChanneldebugFile, "Data: "); fflush(event->sideChanneldebugFile);
                        for (uint32_t i = 0; i < event->header.size; i++) {
                            fprintf(event->sideChanneldebugFile, "%02X ", (int)mm[i]);
                        }
                        fprintf(event->sideChanneldebugFile, "\n"); fflush(event->sideChanneldebugFile);
                    }
                }
                else {
                    // fprintf(globalDebugFile, "%s\t\t\t\tSideChannel error2 %d %x %d\n", timeStamp, event->header.type, devhnd, rc); fflush(globalDebugFile);
                }
                free(mm);
            }
            else if (event->header.type == 555) { //blop message - just ignore
                untilNotDebugMessage = 1;
                // fprintf(globalDebugFile, "%s %x - blop\n", timeStamp, devhnd); fflush(globalDebugFile);
                blopCount++;
            }
        }
        else {
            // fprintf(globalDebugFile, "%s\t\t\t\tSideChannel error3 %d %x %d\n", timeStamp, event->header.type, devhnd, rc); fflush(globalDebugFile);
        }
    } while(untilNotDebugMessage);

    // mvLog(MVLOG_DEBUG,"Incoming event %p: %s %d %p prevEvent: %s %d %p\n",
    //       event,
    //       TypeToStr(event->header.type),
    //       (int)event->header.id,
    //       event->deviceHandle.xLinkFD,
    //       TypeToStr(prevEvent.header.type),
    //       (int)prevEvent.header.id,
    //       prevEvent.deviceHandle.xLinkFD);

    if(rc < 0) {
        // fprintf(globalDebugFile, "%s\t\t\t\tXLink header error %d %x\n", timeStamp, rc, devhnd); fflush(globalDebugFile);
        mvLog(MVLOG_DEBUG,"%s() Read failed %d\n", __func__, (int)rc);
        return rc;
    }

    // if (!blopCount) {
    //     fprintf(globalDebugFile, "%s error! missed a blop message %x\n", timeStamp, devhnd); fflush(globalDebugFile);
    // }

    if (prevEvent->header.id == event->header.id &&
        prevEvent->header.type == event->header.type) {
        fprintf(globalDebugFile, "%s\t\t\t\tR Duplicate id detected %d %x\n", timeStamp, rc, devhnd); fflush(globalDebugFile);
        mvLog(MVLOG_FATAL,"Duplicate id detected. \n");
    }

    *prevEvent = *event;

    return handleIncomingEvent(event, df);
}

//this function should be called only for remote requests
int dispatcherLocalEventGetResponse(xLinkEvent_t* event, xLinkEvent_t* response, FILE *df)
{
    streamDesc_t* stream;
    response->header.id = event->header.id;
    response->header.streamId = event->header.streamId;
    mvLog(MVLOG_DEBUG, "%s\n",TypeToStr(event->header.type));
    
    //fprintf(df, "%s %d\n", __func__, __LINE__);

    switch (event->header.type){
        case XLINK_WRITE_REQ:
        {
            //fprintf(df, "%s %d event is XLINK_WRITE_REQ\n", __func__, __LINE__);
            //in case local tries to write after it issues close (writeSize is zero)
            stream = getStreamById(event->deviceHandle.xLinkFD, event->header.streamId);

            if(!stream) {
                //fprintf(df, "%s %d error stream is NULL\n", __func__, __LINE__);
                mvLog(MVLOG_DEBUG, "stream %d has been closed!\n", event->header.streamId);
                XLINK_SET_EVENT_FAILED_AND_SERVE(event);
                break;
            }

            if (stream->writeSize == 0)
            {
                //fprintf(df, "%s %d error stream writeSize is 0\n", __func__, __LINE__);
                XLINK_EVENT_NOT_ACKNOWLEDGE(event);
                // return -1 to don't even send it to the remote
                releaseStream(stream);
                return -1;
            }
            //fprintf(df, "%s %d event ack\n", __func__, __LINE__);
            XLINK_EVENT_ACKNOWLEDGE(event);
            event->header.flags.bitField.localServe = 0;

            if(!isStreamSpaceEnoughFor(stream, event->header.size)){
                //fprintf(df, "%s %d error isStreamSpaceEnoughFor is not enough\n", __func__, __LINE__);
                mvLog(MVLOG_DEBUG,"local NACK RTS. stream '%s' is full (event %d)\n", stream->name, event->header.id);
                event->header.flags.bitField.block = 1;
                event->header.flags.bitField.localServe = 1;
                // TODO: easy to implement non-blocking read here, just return nack
                mvLog(MVLOG_WARN, "Blocked event would cause dispatching thread to wait on semaphore infinitely\n");
            }else{
                //fprintf(df, "%s %d all good\n", __func__, __LINE__);
                event->header.flags.bitField.block = 0;
                stream->remoteFillLevel += event->header.size;
                // fprintf(globalDebugFile, "Stream %d %s remoteFillLevel++ %lu (now @ %lu)\n", stream->id, stream->name, event->header.size, stream->remoteFillLevel); fflush(globalDebugFile);
                stream->remoteFillPacketLevel++;
                mvLog(MVLOG_DEBUG,"S%d: Got local write of %ld , remote fill level %ld out of %ld %ld\n",
                      event->header.streamId, event->header.size, stream->remoteFillLevel, stream->writeSize, stream->readSize);
            }
            releaseStream(stream);
            //fprintf(df, "%s %d done processing XLINK_WRITE_REQ event\n", __func__, __LINE__);
            break;
        }
        case XLINK_READ_REQ:
        {
            //fprintf(df, "%s %d event is XLINK_READ_REQ\n", __func__, __LINE__);
            stream = getStreamById(event->deviceHandle.xLinkFD, event->header.streamId);
            if(!stream) {
                //fprintf(df, "%s %d error stream is NULL\n", __func__, __LINE__);
                mvLog(MVLOG_DEBUG, "stream %d has been closed!\n", event->header.streamId);
                XLINK_SET_EVENT_FAILED_AND_SERVE(event);
                break;
            }
            streamPacketDesc_t* packet = getPacketFromStream(stream);
            if (packet){
                //fprintf(df, "%s %d event is served with this packet %p\n", __func__, __LINE__, packet);
                //the read can be served with this packet
                event->data = packet;
                XLINK_EVENT_ACKNOWLEDGE(event);
                event->header.flags.bitField.block = 0;
            }
            else{
                //fprintf(df, "%s %d event marked as blocked\n", __func__, __LINE__);
                event->header.flags.bitField.block = 1;
                // TODO: easy to implement non-blocking read here, just return nack
            }
            event->header.flags.bitField.localServe = 1;
            releaseStream(stream);
            //fprintf(df, "%s %d done processing XLINK_READ_REQ event\n", __func__, __LINE__);
            break;
        }
        case XLINK_READ_REL_REQ:
        {
            //fprintf(df, "%s %d event is XLINK_READ_REL_REQ\n", __func__, __LINE__);
            stream = getStreamById(event->deviceHandle.xLinkFD, event->header.streamId);
            if (!stream) {
                //fprintf(df, "%s %d error stream is NULL\n", __func__, __LINE__);
            }
            ASSERT_XLINK(stream);
            XLINK_EVENT_ACKNOWLEDGE(event);
            uint32_t releasedSize = 0;
            releasePacketFromStream(stream, &releasedSize, df);
            event->header.size = releasedSize;
            releaseStream(stream);
            //fprintf(df, "%s %d done processing XLINK_READ_REL_REQ event\n", __func__, __LINE__);
            break;
        }
        case XLINK_READ_REL_SPEC_REQ:
        {
            //fprintf(df, "%s %d event is XLINK_READ_REL_SPEC_REQ\n", __func__, __LINE__);
            uint8_t* data = (uint8_t*)event->data;
            stream = getStreamById(event->deviceHandle.xLinkFD, event->header.streamId);
            ASSERT_XLINK(stream);
            XLINK_EVENT_ACKNOWLEDGE(event);
            uint32_t releasedSize = 0;
            releaseSpecificPacketFromStream(stream, &releasedSize, data, df);
            event->header.size = releasedSize;
            releaseStream(stream);
            //fprintf(df, "%s %d done processing XLINK_READ_REL_SPEC_REQ event\n", __func__, __LINE__);
            break;
        }
        case XLINK_CREATE_STREAM_REQ:
        {
            //fprintf(df, "%s %d event is XLINK_CREATE_STREAM_REQ\n", __func__, __LINE__);
            XLINK_EVENT_ACKNOWLEDGE(event);
#ifdef __PC__
            event->header.streamId = XLinkAddOrUpdateStream(event->deviceHandle.xLinkFD,
                                                            event->header.streamName,
                                                            event->header.size, 0,
                                                            INVALID_STREAM_ID);
            mvLog(MVLOG_DEBUG, "XLINK_CREATE_STREAM_REQ - stream has been just opened with id %ld\n",
                  event->header.streamId);
#else
            mvLog(MVLOG_DEBUG, "XLINK_CREATE_STREAM_REQ - do nothing. Stream will be "
                  "opened with forced id accordingly to response from the host\n");
#endif
            //fprintf(df, "%s %d done processing XLINK_CREATE_STREAM_REQ event\n", __func__, __LINE__);
            break;
        }
        case XLINK_CLOSE_STREAM_REQ:
        {
            //fprintf(df, "%s %d event is XLINK_CLOSE_STREAM_REQ\n", __func__, __LINE__);
            stream = getStreamById(event->deviceHandle.xLinkFD, event->header.streamId);

            ASSERT_XLINK(stream);
            XLINK_EVENT_ACKNOWLEDGE(event);
            if (stream->remoteFillLevel != 0){
                stream->closeStreamInitiated = 1;
                event->header.flags.bitField.block = 1;
                event->header.flags.bitField.localServe = 1;
            }else{
                event->header.flags.bitField.block = 0;
                event->header.flags.bitField.localServe = 0;
            }
            releaseStream(stream);
            //fprintf(df, "%s %d done processing XLINK_CLOSE_STREAM_REQ event\n", __func__, __LINE__);
            break;
        }
        case XLINK_RESET_REQ:
        {
            //fprintf(df, "%s %d event is XLINK_RESET_REQ\n", __func__, __LINE__);
            XLINK_EVENT_ACKNOWLEDGE(event);
            mvLog(MVLOG_DEBUG,"XLINK_RESET_REQ - do nothing\n");
            //fprintf(df, "%s %d done processing XLINK_RESET_REQ event\n", __func__, __LINE__);
            break;
        }
        case XLINK_PING_REQ:
        {
            //fprintf(df, "%s %d event is XLINK_PING_REQ\n", __func__, __LINE__);
            XLINK_EVENT_ACKNOWLEDGE(event);
            mvLog(MVLOG_DEBUG,"XLINK_PING_REQ - do nothing\n");
            //fprintf(df, "%s %d done processing XLINK_PING_REQ event\n", __func__, __LINE__);
            break;
        }
        case XLINK_WRITE_RESP:
        case XLINK_READ_RESP:
        case XLINK_READ_REL_RESP:
        case XLINK_READ_REL_SPEC_RESP:
        case XLINK_CREATE_STREAM_RESP:
        case XLINK_CLOSE_STREAM_RESP:
        case XLINK_PING_RESP:
            break;
        case XLINK_RESET_RESP:
            //should not happen
            event->header.flags.bitField.localServe = 1;
            break;
        default:
        {
            //fprintf(df, "%s %d event is default\n", __func__, __LINE__);
            mvLog(MVLOG_ERROR,
                  "Fail to get response for local event. type: %d, stream name: %s\n",
                  event->header.type, event->header.streamName);
            ASSERT_XLINK(0);
        }
    }
    return 0;
}

//this function should be called only for remote requests
int dispatcherRemoteEventGetResponse(xLinkEvent_t* event, xLinkEvent_t* response, FILE *df)
{
    streamDesc_t* stream;
    response->header.id = event->header.id;
    response->header.streamId = event->header.streamId;
    response->header.flags.raw = 0;
    mvLog(MVLOG_DEBUG, "%s\n",TypeToStr(event->header.type));
    
    //fprintf(df, "%s %d\n", __func__, __LINE__);

    switch (event->header.type)
    {
        case XLINK_WRITE_REQ:
            {
                //fprintf(df, "%s %d\n event is XLINK_WRITE_REQ\n", __func__, __LINE__);
                //let remote write immediately as we have a local buffer for the data
                response->header.type = XLINK_WRITE_RESP;
                response->header.size = event->header.size;
                response->header.streamId = event->header.streamId;
                response->deviceHandle = event->deviceHandle;
                XLINK_EVENT_ACKNOWLEDGE(response);

                // we got some data. We should unblock a blocked read
                int xxx = DispatcherUnblockEvent(-1,
                                                XLINK_READ_REQ,
                                                response->header.streamId,
                                                event->deviceHandle.xLinkFD);
                (void) xxx;
                mvLog(MVLOG_DEBUG,"unblocked from stream %d %d\n",
                    (int)response->header.streamId, (int)xxx);
                //fprintf(df, "%s %d\n done processing XLINK_WRITE_REQ\n", __func__, __LINE__);
            }
            break;
        case XLINK_READ_REQ:
                //fprintf(df, "%s %d\n event is XLINK_READ_REQ - done\n", __func__, __LINE__);
            break;
        case XLINK_READ_REL_SPEC_REQ:
            //fprintf(df, "%s %d\n event is XLINK_READ_REL_SPEC_REQ\n", __func__, __LINE__);
            XLINK_EVENT_ACKNOWLEDGE(response);
            response->header.type = XLINK_READ_REL_SPEC_RESP;
            response->deviceHandle = event->deviceHandle;
            stream = getStreamById(event->deviceHandle.xLinkFD,
                                   event->header.streamId);
            ASSERT_XLINK(stream);
            stream->remoteFillLevel -= event->header.size;
            // fprintf(globalDebugFile, "Stream %d %s remoteFillLevel-- XLINK_READ_REL_SPEC_RESP from  %lu (now @ %lu)\n", stream->id, stream->name, event->header.size, stream->remoteFillLevel); fflush(globalDebugFile);
            stream->remoteFillPacketLevel--;

            mvLog(MVLOG_DEBUG,"S%d: Got remote release of %ld, remote fill level %ld out of %ld %ld\n",
                  event->header.streamId, event->header.size, stream->remoteFillLevel, stream->writeSize, stream->readSize);
            releaseStream(stream);

            DispatcherUnblockEvent(-1, XLINK_WRITE_REQ, event->header.streamId,
                                   event->deviceHandle.xLinkFD);
            //with every released packet check if the stream is already marked for close
            if (stream->closeStreamInitiated && stream->localFillLevel == 0)
            {
                mvLog(MVLOG_DEBUG,"%s() Unblock close STREAM\n", __func__);
                DispatcherUnblockEvent(-1,
                                       XLINK_CLOSE_STREAM_REQ,
                                       event->header.streamId,
                                       event->deviceHandle.xLinkFD);
            }
            //fprintf(df, "%s %d\n done processing XLINK_READ_REL_SPEC_REQ\n", __func__, __LINE__);
            break;
        case XLINK_READ_REL_REQ:
            //fprintf(df, "%s %d\n event is XLINK_READ_REL_REQ\n", __func__, __LINE__);
            XLINK_EVENT_ACKNOWLEDGE(response);
            response->header.type = XLINK_READ_REL_RESP;
            response->deviceHandle = event->deviceHandle;
            stream = getStreamById(event->deviceHandle.xLinkFD,
                                   event->header.streamId);
            ASSERT_XLINK(stream);
            stream->remoteFillLevel -= event->header.size;
            // fprintf(globalDebugFile, "Stream %d %s remoteFillLevel-- XLINK_READ_REL_REQ from  %lu (now @ %lu)\n", stream->id, stream->name, event->header.size, stream->remoteFillLevel); fflush(globalDebugFile);
            stream->remoteFillPacketLevel--;

            mvLog(MVLOG_DEBUG,"S%d: Got remote release of %ld, remote fill level %ld out of %ld %ld\n",
                  event->header.streamId, event->header.size, stream->remoteFillLevel, stream->writeSize, stream->readSize);
            releaseStream(stream);

            DispatcherUnblockEvent(-1, XLINK_WRITE_REQ, event->header.streamId,
                                   event->deviceHandle.xLinkFD);
            //with every released packet check if the stream is already marked for close
            if (stream->closeStreamInitiated && stream->localFillLevel == 0)
            {
                mvLog(MVLOG_DEBUG,"%s() Unblock close STREAM\n", __func__);
                int xxx = DispatcherUnblockEvent(-1,
                                                 XLINK_CLOSE_STREAM_REQ,
                                                 event->header.streamId,
                                                 event->deviceHandle.xLinkFD);
                (void) xxx;
            }
            //fprintf(df, "%s %d\n done processing XLINK_READ_REL_REQ\n", __func__, __LINE__);
            break;
        case XLINK_CREATE_STREAM_REQ:
            //fprintf(df, "%s %d\n event is XLINK_CREATE_STREAM_REQ\n", __func__, __LINE__);
            XLINK_EVENT_ACKNOWLEDGE(response);
            response->header.type = XLINK_CREATE_STREAM_RESP;
            //write size from remote means read size for this peer
#ifndef __PC__
            response->header.streamId = XLinkAddOrUpdateStream(event->deviceHandle.xLinkFD,
                                                               event->header.streamName,
                                                               0, event->header.size,
                                                               event->header.streamId);
#else
            response->header.streamId = XLinkAddOrUpdateStream(event->deviceHandle.xLinkFD,
                                                               event->header.streamName,
                                                               0, event->header.size,
                                                               INVALID_STREAM_ID);
#endif
            if (response->header.streamId == INVALID_STREAM_ID) {
                response->header.flags.bitField.ack = 0;
                response->header.flags.bitField.sizeTooBig = 1;
                break;
            }

            response->deviceHandle = event->deviceHandle;
            mv_strncpy(response->header.streamName, MAX_STREAM_NAME_LENGTH,
                       event->header.streamName, MAX_STREAM_NAME_LENGTH - 1);
            response->header.size = event->header.size;
            mvLog(MVLOG_DEBUG,"creating stream %x\n", (int)response->header.streamId);
            //fprintf(df, "%s %d\n done processing XLINK_CREATE_STREAM_REQ\n", __func__, __LINE__);
            break;
        case XLINK_CLOSE_STREAM_REQ:
        {
            //fprintf(df, "%s %d\n event is XLINK_CLOSE_STREAM_REQ\n", __func__, __LINE__);
            response->header.type = XLINK_CLOSE_STREAM_RESP;
            response->header.streamId = event->header.streamId;
            response->deviceHandle = event->deviceHandle;

            streamDesc_t* stream = getStreamById(event->deviceHandle.xLinkFD,
                                                 event->header.streamId);
            if (!stream) {
                //if we have sent a NACK before, when the event gets unblocked
                //the stream might already be unavailable
                XLINK_EVENT_ACKNOWLEDGE(response);
                mvLog(MVLOG_DEBUG,"%s() got a close stream on aready closed stream\n", __func__);
            } else {
                if (stream->localFillLevel == 0)
                {
                    XLINK_EVENT_ACKNOWLEDGE(response);

                    if (stream->readSize)
                    {
                        stream->readSize = 0;
                        stream->closeStreamInitiated = 0;
                    }

                    if (!stream->writeSize) {
                        stream->id = INVALID_STREAM_ID;
                        stream->name[0] = '\0';
                    }
#ifndef __PC__
                    if(XLink_sem_destroy(&stream->sem))
                        perror("Can't destroy semaphore");
#endif
                }
                else
                {
                    mvLog(MVLOG_DEBUG,"%s():fifo is NOT empty returning NACK \n", __func__);
                    XLINK_EVENT_NOT_ACKNOWLEDGE(response);
                    stream->closeStreamInitiated = 1;
                }

                releaseStream(stream);
                //fprintf(df, "%s %d\n done processing XLINK_CLOSE_STREAM_REQ\n", __func__, __LINE__);
            }
            break;
        }
        case XLINK_PING_REQ:
            //fprintf(df, "%s %d\n event is XLINK_PING_REQ\n", __func__, __LINE__);
            response->header.type = XLINK_PING_RESP;
            XLINK_EVENT_ACKNOWLEDGE(response);
            response->deviceHandle = event->deviceHandle;
            sem_post(&pingSem);
            //fprintf(df, "%s %d\n done processing XLINK_PING_REQ\n", __func__, __LINE__);
            break;
        case XLINK_RESET_REQ:
            //fprintf(df, "%s %d\n event is XLINK_RESET_REQ\n", __func__, __LINE__);
            mvLog(MVLOG_DEBUG,"reset request - received! Sending ACK *****\n");
            XLINK_EVENT_ACKNOWLEDGE(response);
            response->header.type = XLINK_RESET_RESP;
            response->deviceHandle = event->deviceHandle;
            // need to send the response, serve the event and then reset
            //fprintf(df, "%s %d\n done processing XLINK_RESET_REQ\n", __func__, __LINE__);
            break;
        case XLINK_WRITE_RESP:
            //fprintf(df, "%s %d\n event is XLINK_WRITE_RESP - done\n", __func__, __LINE__);
            break;
        case XLINK_READ_RESP:
            //fprintf(df, "%s %d\n event is XLINK_READ_RESP - done\n", __func__, __LINE__);
            break;
        case XLINK_READ_REL_RESP:
            //fprintf(df, "%s %d\n event is XLINK_READ_REL_RESP - done\n", __func__, __LINE__);
            break;
        case XLINK_READ_REL_SPEC_RESP:
            //fprintf(df, "%s %d\n event is XLINK_READ_REL_SPEC_RESP - done\n", __func__, __LINE__);
            break;
        case XLINK_CREATE_STREAM_RESP:
        {
            //fprintf(df, "%s %d\n event is XLINK_CREATE_STREAM_RESP - done\n", __func__, __LINE__);
            // write_size from the response the size of the buffer from the remote
#ifndef __PC__
            response->header.streamId = XLinkAddOrUpdateStream(event->deviceHandle.xLinkFD,
                                                               event->header.streamName,
                                                               event->header.size, 0,
                                                               event->header.streamId);
            XLINK_RET_IF(response->header.streamId
                == INVALID_STREAM_ID);
            mvLog(MVLOG_DEBUG, "XLINK_CREATE_STREAM_REQ - stream has been just opened "
                  "with forced id=%ld accordingly to response from the host\n",
                  response->header.streamId);
#endif
            response->deviceHandle = event->deviceHandle;
            //fprintf(df, "%s %d\n done processing XLINK_CREATE_STREAM_RESP\n", __func__, __LINE__);
            break;
        }
        case XLINK_CLOSE_STREAM_RESP:
        {
            //fprintf(df, "%s %d\n event is XLINK_CLOSE_STREAM_RESP - done\n", __func__, __LINE__);
            streamDesc_t* stream = getStreamById(event->deviceHandle.xLinkFD,
                                                 event->header.streamId);

            if (!stream){
                XLINK_EVENT_NOT_ACKNOWLEDGE(response);
                break;
            }
            stream->writeSize = 0;
            if (!stream->readSize) {
                XLINK_EVENT_NOT_ACKNOWLEDGE(response);
                stream->id = INVALID_STREAM_ID;
                stream->name[0] = '\0';
                break;
            }
            releaseStream(stream);
            //fprintf(df, "%s %d\n done processing XLINK_CLOSE_STREAM_RESP\n", __func__, __LINE__);
            break;
        }
        case XLINK_PING_RESP:
            //fprintf(df, "%s %d\n event is XLINK_PING_RESP - done\n", __func__, __LINE__);
            break;
        case XLINK_RESET_RESP:
            //fprintf(df, "%s %d\n event is XLINK_RESET_RESP - done\n", __func__, __LINE__);
            break;
        default:
        {
            //fprintf(df, "%s %d\n event is default - done\n", __func__, __LINE__);
            mvLog(MVLOG_ERROR,
                "Fail to get response for remote event. type: %d, stream name: %s\n",
                event->header.type, event->header.streamName);
            ASSERT_XLINK(0);
        }
    }
    return 0;
}

void dispatcherCloseLink(void* fd, int fullClose, FILE *df)
{
    xLinkDesc_t* link = getLink(fd);
    if (!link) {
        mvLog(MVLOG_WARN, "Dispatcher link is null");
        return;
    }
    //fprintf(df, "%s %d\n", __func__, __LINE__);

    if (!fullClose) {
        link->peerState = XLINK_DOWN;
        return;
    }

    link->id = INVALID_LINK_ID;
    link->deviceHandle.xLinkFD = NULL;
    link->peerState = XLINK_NOT_INIT;
    link->nextUniqueStreamId = 0;

    for (int index = 0; index < XLINK_MAX_STREAMS; index++) {
        streamDesc_t* stream = &link->availableStreams[index];
        if (!stream) {
            continue;
        }

        while (getPacketFromStream(stream) || stream->blockedPackets) {
            releasePacketFromStream(stream, NULL, df);
        }

        XLinkStreamReset(stream);
    }

    if(XLink_sem_destroy(&link->dispatcherClosedSem)) {
        mvLog(MVLOG_DEBUG, "Cannot destroy dispatcherClosedSem\n");
    }
}

void dispatcherCloseDeviceFd(xLinkDeviceHandle_t* deviceHandle)
{
    XLinkPlatformCloseRemote(deviceHandle);
}

// ------------------------------------
// XLinkDispatcherImpl.h implementation. End.
// ------------------------------------



// ------------------------------------
// Helpers implementation. Begin.
// ------------------------------------

int isStreamSpaceEnoughFor(streamDesc_t* stream, uint32_t size)
{
    int cond1 = stream->remoteFillPacketLevel >= XLINK_MAX_PACKETS_PER_STREAM;
    int cond2 = stream->remoteFillLevel + size > stream->writeSize;
    if(cond1 || cond2) {
        // fprintf(stream->df, "!!! Not enough space in stream %d %s\n", stream->id, stream->name);
        fprintf(globalDebugFile, "!!! S%d: Not enough space in stream '%s' for %lu: PKT %lu, FILL %ld SIZE %lu (%d) cond1-%d cond2-%d\n",
              stream->id, stream->name, size, stream->remoteFillPacketLevel, stream->remoteFillLevel, stream->writeSize, stream->writeSize, cond1, cond2); fflush(globalDebugFile);

        // mvLog(MVLOG_DEBUG, "S%d: Not enough space in stream '%s' for %ld: PKT %ld, FILL %ld SIZE %ld\n",
            //   stream->id, stream->name, size, stream->remoteFillPacketLevel, stream->remoteFillLevel, stream->writeSize);
        return 0;
    }

    return 1;
}

streamPacketDesc_t* getPacketFromStream(streamDesc_t* stream)
{
    streamPacketDesc_t* ret = NULL;
    if (stream->availablePackets)
    {
        ret = &stream->packets[stream->firstPacketUnused];
        stream->availablePackets--;
        CIRCULAR_INCREMENT(stream->firstPacketUnused,
                           XLINK_MAX_PACKETS_PER_STREAM);
        stream->blockedPackets++;
    }
    return ret;
}

int releasePacketFromStream(streamDesc_t* stream, uint32_t* releasedSize, FILE *df)
{
    //fprintf(df, "%s %d\n", __func__, __LINE__);
    streamPacketDesc_t* currPack = &stream->packets[stream->firstPacket];
    if(stream->blockedPackets == 0){\
        //fprintf(df, "%s %d error ! There is no packet to release\n", __func__, __LINE__);
        mvLog(MVLOG_ERROR,"There is no packet to release\n");
        return 0; // ignore this, although this is a big problem on application side
    }

    stream->localFillLevel -= currPack->length;
    mvLog(MVLOG_DEBUG, "S%d: Got release of %ld , current local fill level is %ld out of %ld %ld\n",
          stream->id, currPack->length, stream->localFillLevel, stream->readSize, stream->writeSize);

    XLinkPlatformDeallocateData(currPack->data,
                                ALIGN_UP_INT32((int32_t) currPack->length, __CACHE_LINE_SIZE), __CACHE_LINE_SIZE);

    CIRCULAR_INCREMENT(stream->firstPacket, XLINK_MAX_PACKETS_PER_STREAM);
    stream->blockedPackets--;
    if (releasedSize) {
        *releasedSize = currPack->length;
    }
    return 0;
}

int releaseSpecificPacketFromStream(streamDesc_t* stream, uint32_t* releasedSize, uint8_t* data, FILE *df) {
    //fprintf(df, "%s %d\n", __func__, __LINE__);

    if (stream->blockedPackets == 0) {
        //fprintf(df, "%s %d error There is no packet to release\n", __func__, __LINE__);
        mvLog(MVLOG_ERROR,"There is no packet to release\n");
        return 0; // ignore this, although this is a big problem on application side
    }

    uint32_t packetId = stream->firstPacket;
    uint32_t found = 0;
    do {
        if (stream->packets[packetId].data == data) {
            found = 1;
            break;
        }
        CIRCULAR_INCREMENT(packetId, XLINK_MAX_PACKETS_PER_STREAM);
    } while (packetId != stream->firstPacketUnused);
    ASSERT_XLINK(found);

    streamPacketDesc_t* currPack = &stream->packets[packetId];
    if (currPack->length == 0) {
        //fprintf(df, "%s %d error Packet with ID %d is empty\n", __func__, __LINE__, packetId);
        mvLog(MVLOG_ERROR, "Packet with ID %d is empty\n", packetId);
    }

    stream->localFillLevel -= currPack->length;

  mvLog(MVLOG_DEBUG, "S%d: Got release of %ld , current local fill level is %ld out of %ld %ld\n",
          stream->id, currPack->length, stream->localFillLevel, stream->readSize, stream->writeSize);
    XLinkPlatformDeallocateData(currPack->data,
                                ALIGN_UP_INT32((int32_t) currPack->length, __CACHE_LINE_SIZE), __CACHE_LINE_SIZE);
    stream->blockedPackets--;
    if (releasedSize) {
        *releasedSize = currPack->length;
    }

    if (packetId != stream->firstPacket) {
        uint32_t currIndex = packetId;
        uint32_t nextIndex = currIndex;
        CIRCULAR_INCREMENT(nextIndex, XLINK_MAX_PACKETS_PER_STREAM);
        while (currIndex != stream->firstPacketFree) {
            stream->packets[currIndex] = stream->packets[nextIndex];
            currIndex = nextIndex;
            CIRCULAR_INCREMENT(nextIndex, XLINK_MAX_PACKETS_PER_STREAM);
        }
        CIRCULAR_DECREMENT(stream->firstPacketUnused, (XLINK_MAX_PACKETS_PER_STREAM - 1));
        CIRCULAR_DECREMENT(stream->firstPacketFree, (XLINK_MAX_PACKETS_PER_STREAM - 1));

    } else {
        CIRCULAR_INCREMENT(stream->firstPacket, XLINK_MAX_PACKETS_PER_STREAM);
    }

    return 0;
}

int addNewPacketToStream(streamDesc_t* stream, void* buffer, uint32_t size, FILE *df) {
    //fprintf(df, "%s %d\n", __func__, __LINE__);
    if (stream->availablePackets + stream->blockedPackets < XLINK_MAX_PACKETS_PER_STREAM)
    {
        stream->packets[stream->firstPacketFree].data = buffer;
        stream->packets[stream->firstPacketFree].length = size;
        CIRCULAR_INCREMENT(stream->firstPacketFree, XLINK_MAX_PACKETS_PER_STREAM);
        stream->availablePackets++;
        return 0;
    }
    return -1;
}

int handleIncomingEvent(xLinkEvent_t* event, FILE *df) {
    //this function will be dependent whether this is a client or a Remote
    //specific actions to this peer
    mvLog(MVLOG_DEBUG, "%s, size %u, streamId %u.\n", TypeToStr(event->header.type), event->header.size, event->header.streamId);

    // ASSERT_XLINK(event->header.type >= XLINK_WRITE_REQ
            //    && event->header.type != XLINK_REQUEST_LAST
            //    && event->header.type < XLINK_RESP_LAST);

    uint32_t devhnd = (uint32_t)(event->deviceHandle.xLinkFD) & 0xFFFFF;
    struct timespec ts; //timespec_get(&ts, TIME_UTC);
    char buff[300]; char timeStamp[300];
    clock_gettime(CLOCK_REALTIME, &ts);
    strftime(buff, sizeof buff, "%T", gmtime(&ts.tv_sec));
    sprintf(timeStamp, "%s.%06ld", buff, ts.tv_nsec / 1000);

    switch(event->header.type) {
        case XLINK_WRITE_REQ:          fprintf(globalDebugFile, "%s\t\t\t\tR  %x %s ch_%d id_%d %d\n", timeStamp, devhnd, T2S(event->header.type), event->header.streamId, event->header.id, event->header.size); fflush(globalDebugFile); break;
        case XLINK_READ_REQ:           fprintf(globalDebugFile, "%s\t\t\t\tR  %x %s ch_%d id_%d %d\n", timeStamp, devhnd, T2S(event->header.type), event->header.streamId, event->header.id, event->header.size); fflush(globalDebugFile); break;
        case XLINK_READ_REL_REQ:       fprintf(globalDebugFile, "%s\t\t\t\tR  %x %s ch_%d id_%d %d\n", timeStamp, devhnd, T2S(event->header.type), event->header.streamId, event->header.id, event->header.size); fflush(globalDebugFile); break;
        case XLINK_READ_REL_SPEC_REQ:  fprintf(globalDebugFile, "%s\t\t\t\tR  %x %s ch_%d id_%d %d\n", timeStamp, devhnd, T2S(event->header.type), event->header.streamId, event->header.id, event->header.size); fflush(globalDebugFile); break;
        case XLINK_CREATE_STREAM_REQ:  fprintf(globalDebugFile, "%s\t\t\t\tR  %x %s id_%d %s\n",       timeStamp, devhnd, T2S(event->header.type), event->header.id, event->header.streamName); fflush(globalDebugFile); break;
        case XLINK_CLOSE_STREAM_REQ:   fprintf(globalDebugFile, "%s\t\t\t\tR  %x %s id_%d %s\n",       timeStamp, devhnd, T2S(event->header.type), event->header.id, event->header.streamName); fflush(globalDebugFile); break;
        case XLINK_PING_REQ:           fprintf(globalDebugFile, "%s\t\t\t\tR  %x %s id_%d \n",         timeStamp, devhnd, T2S(event->header.type), event->header.id); fflush(globalDebugFile); break;
        case XLINK_RESET_REQ:          fprintf(globalDebugFile, "%s\t\t\t\tR  %x %s id_%d \n",         timeStamp, devhnd, T2S(event->header.type), event->header.id); fflush(globalDebugFile); break;
        case XLINK_REQUEST_LAST:       fprintf(globalDebugFile, "%s\t\t\t\tR  %x %s ch_%d id_%d \n",   timeStamp, devhnd, T2S(event->header.type), event->header.streamId, event->header.id); fflush(globalDebugFile); break;
        case XLINK_WRITE_RESP:         fprintf(globalDebugFile, "%s\t\t\t\tR  %x %s ch_%d id_%d %d\n", timeStamp, devhnd, T2S(event->header.type), event->header.streamId, event->header.id, event->header.size); fflush(globalDebugFile); break;
        case XLINK_READ_RESP:          fprintf(globalDebugFile, "%s\t\t\t\tR  %x %s ch_%d id_%d %d\n", timeStamp, devhnd, T2S(event->header.type), event->header.streamId, event->header.id, event->header.size); fflush(globalDebugFile); break;
        case XLINK_READ_REL_RESP:      fprintf(globalDebugFile, "%s\t\t\t\tR  %x %s ch_%d id_%d %d\n", timeStamp, devhnd, T2S(event->header.type), event->header.streamId, event->header.id, event->header.size); fflush(globalDebugFile); break;
        case XLINK_READ_REL_SPEC_RESP: fprintf(globalDebugFile, "%s\t\t\t\tR  %x %s ch_%d id_%d %d\n", timeStamp, devhnd, T2S(event->header.type), event->header.streamId, event->header.id, event->header.size); fflush(globalDebugFile); break;
        case XLINK_CREATE_STREAM_RESP: fprintf(globalDebugFile, "%s\t\t\t\tR  %x %s id_%d %s\n",       timeStamp, devhnd, T2S(event->header.type), event->header.id, event->header.streamName); fflush(globalDebugFile); break;
        case XLINK_CLOSE_STREAM_RESP:  fprintf(globalDebugFile, "%s\t\t\t\tR  %x %s id_%d %s\n",       timeStamp, devhnd, T2S(event->header.type), event->header.id, event->header.streamName); fflush(globalDebugFile); break;
        case XLINK_PING_RESP:          fprintf(globalDebugFile, "%s\t\t\t\tR  %x %s id_%d \n",         timeStamp, devhnd, T2S(event->header.type), event->header.id); fflush(globalDebugFile); break;
        case XLINK_RESET_RESP:         fprintf(globalDebugFile, "%s\t\t\t\tR  %x %s id_%d \n",         timeStamp, devhnd, T2S(event->header.type), event->header.id); fflush(globalDebugFile); break;
        case XLINK_RESP_LAST:          fprintf(globalDebugFile, "%s\t\t\t\tR  %x %s id_%d \n",         timeStamp, devhnd, T2S(event->header.type), event->header.id); fflush(globalDebugFile); break;
        default:                       fprintf(globalDebugFile, "%s\t\t\t\tR  %x %s id_%d \n",         timeStamp, devhnd, T2S(event->header.type), event->header.id); fflush(globalDebugFile); break;
    };


    // Then read the data buffer, which is contained only in the XLINK_WRITE_REQ event
    if(event->header.type != XLINK_WRITE_REQ) {
         return 0;
    }

    int rc = -1;
    streamDesc_t* stream = getStreamById(event->deviceHandle.xLinkFD, event->header.streamId);
    ASSERT_XLINK(stream);

    stream->localFillLevel += event->header.size;
    mvLog(MVLOG_DEBUG,"S%d: Got write of %ld, current local fill level is %ld out of %ld %ld\n",
          event->header.streamId, event->header.size, stream->localFillLevel, stream->readSize, stream->writeSize);

    void* buffer = XLinkPlatformAllocateData(ALIGN_UP(ROUND_UP(event->header.size, USB_SIZE_MULTIPLE), __CACHE_LINE_SIZE), __CACHE_LINE_SIZE);
    XLINK_OUT_WITH_LOG_IF(buffer == NULL,
        mvLog(MVLOG_FATAL,"out of memory to receive data of size = %zu\n", event->header.size));

    const int sc = XLinkPlatformRead(&event->deviceHandle, buffer, event->header.size);

    clock_gettime(CLOCK_REALTIME, &ts);
    strftime(buff, sizeof buff, "%T", gmtime(&ts.tv_sec));
    sprintf(timeStamp, "%s.%06ld", buff, ts.tv_nsec / 1000);


    if (sc == 0) {
        if (event->header.size == 80200) {
            uint8_t *data = (uint8_t*)buffer;
            uint32_t *final_scores = (uint32_t*)(data + 80000);
            uint32_t *final_is_frontal_scores = (uint32_t*)(data + 80064);
            uint32_t *final_boxes = (uint32_t*)(data + 80128);
            uint32_t *final_face_boxes = (uint32_t*)(data + 80192);
            float *bboxes = (float*)(data);
            float *fboxes = (float*)(data + 32000);
            float *frontal = (float*)(data + 64000);
            float *scores = (float*)(data + 72000);
            // char fname[300];
            // sprintf(fname, "FD_out_%d.bin", FD_output_count % 30);

            if (final_scores[0] != final_is_frontal_scores[0] ||
                final_scores[0] != final_boxes[0] ||
                final_scores[0] != final_face_boxes[0] ||
                final_scores[1] != 1 ||
                final_is_frontal_scores[1] != 1 ||
                final_boxes[1] != 4 ||
                final_face_boxes[1] != 4) {
                fprintf(globalDebugFile, "%s\t\t\t\t%x FD output #%u - error for FD output sizes %x %x - %x %x - %x %x - %x %x\n", timeStamp, devhnd, FD_output_count,
                    final_scores[0], final_scores[1], 
                    final_is_frontal_scores[0], final_is_frontal_scores[1], 
                    final_boxes[0], final_boxes[1], 
                    final_face_boxes[0], final_face_boxes[1]); fflush(globalDebugFile);
            }

            uint32_t max2Process = final_boxes[0];
            if (max2Process > 10) max2Process = 10;
            char z[2048+512];
            int pos = 0;
            pos = sprintf(z, "%x ", final_boxes[0]);
            for (int i = 0; i < max2Process; i++) {
                pos += sprintf(z + pos, "%.2f %.2f(%.2f %.2f %.2f %.2f)(%.2f %.2f %.2f %.2f)",
                scores[i], frontal[i],
                bboxes[4 * i + 0], bboxes[4 * i + 1], bboxes[4 * i + 2], bboxes[4 * i + 3],
                fboxes[4 * i + 0], fboxes[4 * i + 1], fboxes[4 * i + 2], fboxes[4 * i + 3]);
            }
            
            fprintf(globalDebugFile, "%s\t\t\t\t%x FD output #%u id_%d faces%s\n", timeStamp, devhnd, FD_output_count++, event->header.id, z); fflush(globalDebugFile);
            
            // FILE *fout;
            // fout = fopen(fname, "wb");
            // if (fout != NULL) {
            //     fwrite(buffer, sizeof(uint8_t), event->header.size, fout);
            //     fclose(fout);
            // }         
        }
        else if (event->header.size == 114704)
        {
            fprintf(globalDebugFile, "%s\t\t\t\t%x ASD output #%u id_%d\n", timeStamp, devhnd, ASD_ouput_count++, event->header.id); fflush(globalDebugFile);
        }
        else /*if (event->header.size > 1024)*/ {
            fprintf(globalDebugFile, "%s\t\t\t\tR+ %x %s ch_%d id_%d s_%d\n", timeStamp, devhnd, T2S(event->header.type), event->header.streamId, event->header.id, event->header.size);
        }
    }
    else {
        // fprintf(globalDebugFile, "%s\t\t\t\tXLink data error %d %x %s ch_%d id_%d s_%d\n", timeStamp, sc, devhnd, T2S(event->header.type), event->header.streamId, event->header.id, event->header.size); fflush(globalDebugFile);
    }

    XLINK_OUT_WITH_LOG_IF(sc < 0, mvLog(MVLOG_ERROR,"%s() Read failed %d\n", __func__, sc));

    event->data = buffer;
    XLINK_OUT_WITH_LOG_IF(addNewPacketToStream(stream, buffer, event->header.size, df),
        mvLog(MVLOG_WARN,"No more place in stream. release packet\n"));
    rc = 0;

XLINK_OUT:
    releaseStream(stream);

    if(rc != 0) {
        if(buffer != NULL) {
            XLinkPlatformDeallocateData(buffer,
                ALIGN_UP(event->header.size, __CACHE_LINE_SIZE), __CACHE_LINE_SIZE);
        }
        XLINK_EVENT_NOT_ACKNOWLEDGE(event);
    }

    return rc;
}

// ------------------------------------
// Helpers implementation. Begin.
// ------------------------------------
