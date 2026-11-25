use std::io::{self, Write};
use std::time::Duration;

use plait_core::serving::{
    DecoderCheckpoint, DecoderOffer, DecoderStep, RouteSetId, ServingAction, ServingLookup,
    ServingView,
};
use plait_ops::decoder::LinearProjection;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::net::TcpStream;

use super::{
    Admission, DecodeSession, Generation, GroupControl, Job, RPC, Reply, Request, Worker, apply,
    id, now_ms, read_frame, transfer, write_frame,
};

const CHUNK_SIZE: u32 = 8;
const CHECKPOINT_TOKENS: u32 = 32;
const WRITE_TIMEOUT: Duration = Duration::from_secs(3);

#[derive(Serialize, Deserialize)]
struct Ack {
    next_offset: u32,
    #[serde(default)]
    checkpoint: bool,
}

impl<P: LinearProjection> Worker<'_, P> {
    pub(super) async fn stream_decode(
        &mut self,
        stream: &mut TcpStream,
        job: Job,
        max_new_tokens: u32,
        stop: &[u32],
        mut checkpoint_enabled: bool,
        through_offset: u32,
    ) -> io::Result<Reply> {
        if max_new_tokens == 0 || max_new_tokens == u32::MAX {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid streaming token limit",
            ));
        }
        let DecodeSession {
            mut cache,
            mut first_token,
            owner,
            source,
            mut offset,
        } = match self.claim_decode(job, Some(through_offset)).await? {
            Ok(session) => session,
            Err(reply) => return Ok(reply),
        };
        if offset > max_new_tokens + 1 {
            return Err(invalid("restored KV exceeds the requested output limit"));
        }
        let start = Reply::StreamStart {
            owner,
            offset,
            last_token: first_token,
        };
        tokio::time::timeout(WRITE_TIMEOUT, write_frame(stream, &start)).await??;
        let mut checkpoint_at = offset;
        while offset <= max_new_tokens && !stop.contains(&first_token) {
            let count = (max_new_tokens - offset + 1).min(CHUNK_SIZE);
            let tokens = self
                .head
                .generate(self.stack, &mut cache, &[first_token], count, stop)
                .map_err(io::Error::other)?;
            first_token = *tokens
                .last()
                .expect("a positive generation limit produces tokens");
            let next_offset = offset + tokens.len() as u32;
            let mut checkpoint = None;
            let suffix_tokens = next_offset - checkpoint_at;
            if checkpoint_enabled && suffix_tokens >= CHECKPOINT_TOKENS {
                let start = cache.tokens() - suffix_tokens as usize;
                let payload = transfer::encode_suffix(&cache, start)?;
                match transfer::deposit(source, &payload).await {
                    Ok(source) => {
                        checkpoint = Some(DecoderCheckpoint {
                            source,
                            start_offset: checkpoint_at,
                            end_offset: next_offset,
                            last_token: first_token,
                        })
                    }
                    Err(_) => checkpoint_enabled = false,
                }
            }
            let chunk = Reply::Chunk {
                owner,
                offset,
                tokens,
                checkpoint,
            };
            tokio::time::timeout(WRITE_TIMEOUT, write_frame(stream, &chunk)).await??;
            let ack: Ack = tokio::time::timeout(RPC, read_frame(stream)).await??;
            if ack.next_offset != next_offset || (ack.checkpoint && checkpoint.is_none()) {
                return Err(invalid(
                    "decoder received an invalid stream acknowledgement",
                ));
            }
            if ack.checkpoint {
                checkpoint_at = next_offset;
            }
            offset = next_offset;
        }
        self.advance(
            id(),
            job,
            DecoderStep::FinishDecodeStream {
                owner,
                end_offset: offset,
            },
        )
        .await?;
        Ok(Reply::StreamComplete { end_offset: offset })
    }

    pub(super) async fn restore_checkpoints(
        &mut self,
        job: Job,
        offer: DecoderOffer,
        through: u32,
        snapshot: &mut transfer::Snapshot,
    ) -> (u32, u32) {
        let mut offset = 1;
        let mut last = offer.first_token;
        if through <= offset {
            return (offset, last);
        }
        let lookup = ServingLookup::DecoderCheckpoints {
            model: RouteSetId::for_model(&self.model),
            work: job.work,
            transfer: offer.source.transfer,
        };
        let Ok(ServingView::DecoderCheckpoints(checkpoints)) = self.control.query(lookup).await
        else {
            return (offset, last);
        };
        for checkpoint in checkpoints {
            if checkpoint.end_offset > through
                || checkpoint.start_offset != offset
                || offer
                    .prefix_tokens
                    .saturating_add(checkpoint.end_offset - 1)
                    > self.limits.tokens
            {
                break;
            }
            let Ok(delta) = transfer::pull(
                checkpoint.source,
                self.limits.bytes,
                self.limits.layers,
                Duration::from_secs(2),
            )
            .await
            else {
                break;
            };
            if delta.tokens != checkpoint.end_offset - offset || snapshot.append(delta).is_err() {
                break;
            }
            offset = checkpoint.end_offset;
            last = checkpoint.last_token;
        }
        (offset, last)
    }
}

pub(super) struct Delivery {
    control: GroupControl,
    tokens: Vec<u32>,
    replay: bool,
    detached: bool,
}

impl Delivery {
    pub(super) fn new(control: GroupControl, replay: bool) -> Self {
        Self {
            control,
            tokens: Vec::new(),
            replay,
            detached: false,
        }
    }

    pub(super) fn first(&mut self, token: u32) -> io::Result<()> {
        self.publish(0, &[token])
    }

    pub(super) fn seed_prefix(&mut self, prefix: &[u32], offset: u32) -> io::Result<()> {
        let offset = offset as usize;
        let consumed = prefix
            .get(..offset)
            .ok_or_else(|| invalid("invalid replay offset"))?;
        self.tokens.extend_from_slice(consumed);
        self.publish(offset, &prefix[offset..])
    }

    pub(super) fn event(&mut self, event: &serde_json::Value) -> io::Result<()> {
        if !self.detached {
            let result = write_event(&mut io::stdout().lock(), event);
            self.check_output(result)?;
        }
        Ok(())
    }

    fn publish(&mut self, offset: usize, tokens: &[u32]) -> io::Result<()> {
        if !self.detached {
            let result = accept(&mut self.tokens, &mut io::stdout().lock(), offset, tokens);
            self.check_output(result)?;
            if !self.detached {
                return Ok(());
            }
        }
        accept(&mut self.tokens, &mut io::sink(), offset, tokens)
    }

    fn check_output(&mut self, result: io::Result<()>) -> io::Result<()> {
        match result {
            Err(error) if self.replay && error.kind() == io::ErrorKind::BrokenPipe => {
                self.detached = true;
                Ok(())
            }
            Err(error) if error.kind() != io::ErrorKind::InvalidData => {
                Err(io::Error::new(io::ErrorKind::InvalidInput, error))
            }
            result => result,
        }
    }

    pub(super) fn finish(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.tokens)
    }

    pub(super) async fn receive(
        &mut self,
        model: &str,
        admission: &Admission,
        input: &Generation,
    ) -> io::Result<Reply> {
        tokio::time::timeout(RPC, async {
            let mut stream = TcpStream::connect(admission.route.decode.addr).await?;
            stream.set_nodelay(true)?;
            write_frame(
                &mut stream,
                &Request::DecodeStream {
                    job: admission.job,
                    max_new_tokens: input.max_new_tokens - 1,
                    stop_tokens: input.stop_tokens.clone(),
                    checkpoint: admission.recoverable,
                    through_offset: self.tokens.len() as u32,
                },
            )
            .await?;
            let mut next = 0_u32;
            let mut claim = None;
            loop {
                match read_frame(&mut stream).await? {
                    Reply::StreamStart {
                        owner,
                        offset,
                        last_token,
                    } if claim.is_none() && offset > 0 => {
                        if self.tokens.get(offset as usize - 1) != Some(&last_token) {
                            return Err(invalid(
                                "checkpoint disagrees with the published stream prefix",
                            ));
                        }
                        if offset > 1 {
                            eprintln!("decoder restored KV through output offset {offset}");
                        }
                        next = offset;
                        claim = Some(owner);
                    }
                    Reply::Chunk {
                        owner,
                        offset,
                        tokens,
                        checkpoint,
                    } => {
                        if offset != next
                            || tokens.is_empty()
                            || tokens.len() > CHUNK_SIZE as usize
                            || claim != Some(owner)
                        {
                            return Err(invalid("noncontiguous decoder stream or changed owner"));
                        }
                        let end = next
                            .checked_add(tokens.len() as u32)
                            .filter(|end| *end <= input.max_new_tokens)
                            .ok_or_else(|| invalid("decoder stream exceeds requested limit"))?;
                        let step = if let Some(checkpoint) = checkpoint {
                            if checkpoint.end_offset != end
                                || tokens.last() != Some(&checkpoint.last_token)
                            {
                                return Err(invalid(
                                    "KV checkpoint does not match the stream chunk",
                                ));
                            }
                            DecoderStep::CheckOutputCheckpoint { owner, checkpoint }
                        } else {
                            DecoderStep::CheckOutput { owner }
                        };
                        let now_ms = now_ms()?;
                        let action = if self.replay {
                            ServingAction::AppendDecoderOutput {
                                model: model.to_owned(),
                                work: admission.job.work,
                                path: admission.job.path,
                                grant: admission.route.decode.grant,
                                now_ms,
                                owner,
                                offset,
                                tokens: tokens.clone(),
                                checkpoint,
                            }
                        } else {
                            ServingAction::AdvanceDecoder {
                                model: model.to_owned(),
                                work: admission.job.work,
                                path: admission.job.path,
                                grant: admission.route.decode.grant,
                                now_ms,
                                step,
                            }
                        };
                        apply(&mut self.control, id(), action).await?;
                        self.publish(offset as usize, &tokens)?;
                        next = end;
                        write_frame(
                            &mut stream,
                            &Ack {
                                next_offset: next,
                                checkpoint: checkpoint.is_some(),
                            },
                        )
                        .await?;
                    }
                    Reply::StreamComplete { end_offset }
                        if claim.is_some()
                            && end_offset == next
                            && self.tokens.len() == next as usize =>
                    {
                        return Ok(Reply::StreamComplete { end_offset });
                    }
                    Reply::KvUnavailable { message } => {
                        return Err(io::Error::new(io::ErrorKind::ConnectionAborted, message));
                    }
                    Reply::Error { message } => return Err(io::Error::other(message)),
                    _ => {
                        return Err(invalid(
                            "decoder stream ended before reproducing its published prefix",
                        ));
                    }
                }
            }
        })
        .await?
    }
}

fn accept(
    tokens: &mut Vec<u32>,
    out: &mut impl Write,
    offset: usize,
    chunk: &[u32],
) -> io::Result<()> {
    let existing = tokens
        .get(offset..)
        .ok_or_else(|| invalid("decoder stream has a gap"))?;
    let overlap = existing.len().min(chunk.len());
    if existing[..overlap] != chunk[..overlap] {
        return Err(invalid(
            "replacement decoder disagrees with the published prefix",
        ));
    }
    let new = &chunk[overlap..];
    if new.is_empty() {
        return Ok(());
    }
    write_event(
        out,
        &json!({ "event": "tokens", "offset": tokens.len(), "tokens": new }),
    )?;
    tokens.extend_from_slice(new);
    Ok(())
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

pub(super) fn write_event(out: &mut impl Write, event: &serde_json::Value) -> io::Result<()> {
    let mut line = serde_json::to_vec(event)?;
    line.push(b'\n');
    out.write_all(&line)?;
    out.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_only_publishes_matching_suffix_and_rejects_divergence() {
        let mut output = Vec::new();
        let mut tokens = Vec::new();
        accept(&mut tokens, &mut output, 0, &[1, 2]).unwrap();
        accept(&mut tokens, &mut output, 0, &[1]).unwrap();
        accept(&mut tokens, &mut output, 1, &[2, 3, 4]).unwrap();
        let before = output.clone();
        assert_eq!(
            accept(&mut tokens, &mut output, 2, &[9, 4, 5])
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );
        assert_eq!(output, before);
        assert_eq!(tokens, [1, 2, 3, 4]);
        let lines: Vec<_> = std::str::from_utf8(&output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect();
        assert_eq!(
            lines,
            [
                json!({ "event": "tokens", "offset": 0, "tokens": [1, 2] }),
                json!({ "event": "tokens", "offset": 2, "tokens": [3, 4] }),
            ]
        );
    }
}
