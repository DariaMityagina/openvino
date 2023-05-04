// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef _XLINKDISPATCHERIMPL_H
#define _XLINKDISPATCHERIMPL_H

#include "XLinkPrivateDefines.h"
#include <stdio.h>

int dispatcherEventSend (xLinkEvent_t*, FILE *df);
int dispatcherEventReceive (xLinkEvent_t*, xLinkEvent_t* prevEvent, FILE *df);
int dispatcherLocalEventGetResponse (xLinkEvent_t*,
                        xLinkEvent_t*, FILE *df);
int dispatcherRemoteEventGetResponse (xLinkEvent_t*,
                        xLinkEvent_t*, FILE *df);
void dispatcherCloseLink (void* fd, int fullClose, FILE *df);
void dispatcherCloseDeviceFd (xLinkDeviceHandle_t* deviceHandle);

#endif //_XLINKDISPATCHERIMPL_H
