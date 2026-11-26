---- MODULE PlaitDeviceCompletion ----
\* One output allocation stays live from submit through the host copy.
\* The output write and event signal are separate steps on the same stream.
\* A different session, epoch, operation, shard, or fence may not signal it.
\* Enqueue/order and successful event synchronization implement the driver
\* assumptions. The caller supplies the expected kernel result.

CompletionStamp(session, epoch, operation, shard, fence) ==
    [session |-> session, epoch |-> epoch, operation |-> operation, shard |-> shard, fence |-> fence]
NoStamp == CompletionStamp(0, 0, 0, 0, 0)
EmptyDeviceJob == [phase |-> "Idle", stamp |-> NoStamp, event |-> NoStamp, output |-> 0, snapshot |-> 0]

QueueDevice(stamp) == [EmptyDeviceJob EXCEPT !.phase = "Queued", !.stamp = stamp]
CanWriteDevice(job) == job.phase = "Queued"
WriteDevice(job, result) == [job EXCEPT !.phase = "Written", !.output = result]
CanSignalDevice(job, stamp) == job.phase = "Written" /\ job.stamp = stamp
SignalDevice(job, stamp) == [job EXCEPT !.phase = "Signaled", !.event = stamp]
CanCopyDevice(job, stamp) ==
    job.phase = "Signaled" /\ job.stamp = stamp /\ job.event = stamp
CopyDevice(job) == [job EXCEPT !.phase = "Copied", !.snapshot = job.output]
CanPublishDevice(job, stamp) ==
    job.phase = "Copied" /\ job.stamp = stamp /\ job.event = stamp

DevicePhases == {"Idle", "Queued", "Written", "Signaled", "Copied"}
DeviceJobSound(job, stamp, expected) ==
    /\ job.phase \in DevicePhases
    /\ job.phase # "Idle" => job.stamp = stamp
    /\ job.phase \in {"Written", "Signaled", "Copied"} => job.output = expected
    /\ job.phase \in {"Signaled", "Copied"} => job.event = stamp
    /\ job.phase = "Copied" => job.snapshot = job.output
====
