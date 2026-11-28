#ifndef PLAIT_PEER_H
#define PLAIT_PEER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ncclComm;
typedef struct PlaitPeerListener PlaitPeerListener;
typedef struct PlaitPeerRequest PlaitPeerRequest;

typedef struct {
    uint8_t group[16];
    uint64_t epoch;
} PlaitPeerGroupVersion;

typedef struct {
    uint8_t node[16];
    uint32_t ordinal;
} PlaitPeerDevice;

typedef struct {
    PlaitPeerDevice device;
    uint64_t incarnation;
    uint64_t fence;
} PlaitPeerGrant;

typedef struct {
    PlaitPeerGroupVersion version;
    PlaitPeerGrant grant;
    char address[80];
    uint8_t ticket[16];
} PlaitPeerMember;

enum {
    PLAIT_PEER_CUDA = 0,
    PLAIT_PEER_ROCM = 1,
    PLAIT_PEER_HUAWEI = 2,
    PLAIT_PEER_PPU = 3
};

/* Use a concrete IP and port 0 for an automatically allocated port. */
int32_t plaitPeerListen(const char *address, const PlaitPeerGroupVersion *version,
                        const PlaitPeerGrant *grant, const uint8_t ticket[16],
                        PlaitPeerListener **listener, PlaitPeerMember *member);

/* Before connecting, set the total payload a worker can send in a replicated
   AllReduce. Every member must specify the same value. Zero restricts
   replication to exchanges with zero egress. Omit for the default policy. */
int32_t plaitPeerSetReplicatedEgressBytes(PlaitPeerListener *listener, uint64_t bytes);

/* Accept the full manifest in any order. The join attempt consumes the listener
   and sets its pointer to NULL. Invalid input rejected before the join leaves it
   owned by the caller. The communicator uses the existing NCCL collectives. */
int32_t plaitPeerConnect(PlaitPeerListener **listener, const PlaitPeerMember *members,
                         size_t count, int32_t backend, uint32_t device,
                         uint32_t timeout_ms, struct ncclComm **comm);
int32_t plaitPeerListenerDestroy(PlaitPeerListener *listener);
int32_t plaitPeerDeviceSlot(const struct ncclComm *comm, const PlaitPeerDevice *device,
                            int32_t *slot);
int32_t plaitPeerManifestDigest(const struct ncclComm *comm, uint8_t digest[32]);

/* Cancel outstanding socket I/O. Wait for active calls before destroying the
   communicator with ncclCommDestroy or releasing its buffers. */
int32_t plaitPeerCancel(const struct ncclComm *comm);
int32_t plaitPeerFailed(const struct ncclComm *comm, int32_t *failed);

enum {
    PLAIT_PEER_ALL_REDUCE = 1,
    PLAIT_PEER_ALL_GATHER = 2,
    PLAIT_PEER_REDUCE_SCATTER = 3,
    PLAIT_PEER_BROADCAST = 4,
    PLAIT_PEER_REDUCE = 5
};

enum { PLAIT_PEER_BUSY = 5 };

typedef struct {
    int32_t kind;
    int32_t datatype;   /* NCCL datatype code */
    int32_t reduction;  /* NCCL reduction code for reducing operations */
    int32_t root;       /* Logical slot for broadcast and reduce */
    size_t count;       /* NCCL count: input count for gather, output count for scatter */
} PlaitPeerCollective;

/* Reserve a pending queue slot and host staging for the input and output,
   then stage input from the producer stream and enqueue network work.
   Successful submission owns the staged bytes and returns a request.
   Capacity rejection returns BUSY without touching the device or reserving
   a sequence. Retry or issue the same collective through its blocking entry
   point before issuing later calls. Peers submit matching operations in order. */
int32_t plaitPeerCollectiveSubmit(const struct ncclComm *comm,
                                 const PlaitPeerCollective *operation, const void *send,
                                 void *producer_stream, PlaitPeerRequest **request);

/* Test host completion. Wait reports errors and delivers the result, then
   consumes the request and sets it to NULL. Argument errors leave it live.
   Use the originating communicator, ready destination and its consumer
   stream. Reduce on non-root slots expects zero output bytes. */
int32_t plaitPeerRequestTest(PlaitPeerRequest *request, int32_t *ready);
int32_t plaitPeerRequestWait(const struct ncclComm *comm, PlaitPeerRequest **request,
                            void *recv, size_t recv_bytes, void *consumer_stream);

/* Abandon the result. An accepted operation still occupies its wire sequence.
   Use plaitPeerCancel to interrupt the group's network operations. */
int32_t plaitPeerRequestDestroy(PlaitPeerRequest *request);

#ifdef __cplusplus
}
#endif

#endif
